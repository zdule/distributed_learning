import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.random

import os, sys
import traceback
import threading

from itertools import islice
from types import SimpleNamespace

from utils import Level, print_d, eval_arg

from timing import end_timing_experiment, start_timer, end_timer, writeout_timer
import time
from allreduce import built_in_allreduce, ring_allreduce
from reducers import NodeAgreggateReducerCPU
from ourdist import OurDist, SeqDist, SeqMergeDist
from config import parse_args

def worker_process(node_id, worker_id, config, reducer):
    torch.manual_seed(1234)
    train_set = config.get_partition_dataset(node_id, worker_id, config.node_dev, config.total_dev, config.dataset_root)
    device = config.devices[worker_id]
    model = config.create_network().to(device)
    model = config.distribute_model(model, reducer, config.grouping_size)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    loss_f = open(config.experiment_name+"_loss.txt", "a")
    
    start_time = time.time()
    batch_count = 0
    #while time.time() - start_time < config.duration and epoch_count < config.epoch_count:
    for epoch in range(config.epoch_count):
        gener = islice(train_set, config.limit_batches)
        while True:
        #for data, target in islice(train_set, config.limit_batches):
            print(f"Node {node_id}, worker {worker_id}, batch, {batch_count} time {time.time()}")
            start_timer("batch")
            start_timer("get_data")
            nx = next(gener, None)
            if nx == None: break
            data, target = nx
            end_timer("get_data")
            #if time.time() - start_time > config.duration:
                #break
            batch_count += 1

            print_d("WORKER: Moving training data to device", Level.DEBUG)
            start_timer("data2dev")
            data, target = data.to(device), target.to(device)
            end_timer("data2dev")
            
            start_timer("zero_grad")
            optimizer.zero_grad()
            end_timer("zero_grad")

            # Feed forward
            print_d("WORKER: Perfroming feed forward", Level.DEBUG)
            start_timer("forward")
            output = model(data)
            loss = F.nll_loss(output, target)
            end_timer("forward")

            # Backpropagation
            print_d("WORKER: Perfroming backprop", Level.DEBUG)
            start_timer("backprop")
            loss.backward()
            end_timer("backprop")
   
            start_timer("sync") 
            model.sync_gradients()
            end_timer("sync")

            # Optimizer step 
            start_timer("optimizer_step")
            optimizer.step()
            end_timer("optimizer_step")

            loss_message = f"Worker {node_id}:{worker_id} loss for batch {batch_count}: {loss}"
            print_d(loss_message, Level.DEBUG)
            loss_f.write(loss_message + "\n")
            end_timer("batch")
            print(f"End: Node {node_id}, worker {worker_id}, batch, {batch_count} time {time.time()}")
    end_time = time.time()

    loss_f.close()
    model.cleanup()
    reducer.cleanup()

    end_timing_experiment(config.experiment_name, extra_fields={"throughput" : batch_count/(end_time-start_time), "batch_time" : (end_time-start_time)*1000/batch_count})
    writeout_timer(config.experiment_name+"_times.csv")

def worker_process_wrapper(*args):
    try:
        worker_process(*args)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

def main_ourdist(config):
    config.distribute_model = OurDist
    cpu_reducer = NodeAgreggateReducerCPU(built_in_allreduce, config.node_dev)
    config.experiment_name = "ourdist"

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i, config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_seq(config):
    config.distribute_model = SeqDist
    cpu_reducer = NodeAgreggateReducerCPU(built_in_allreduce, config.node_dev)

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i,config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_seq_merge(config):
    config.distribute_model = SeqMergeDist
    cpu_reducer = NodeAgreggateReducerCPU(built_in_allreduce, config.node_dev)
    config.experiment_name = "seq_merge"

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i,config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_overlap(config):
    config.distribute_model = OurDist
    cpu_reducer = NodeAgreggateReducerCPU(built_in_allreduce, config.node_dev)
    config.experiment_name = "overlap"
    old_grouping = config.grouping_size
    config.grouping_size = 0

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i, config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()
    config.grouping_size = old_grouping

def main_ddp(config):
    def distribute_model(model, reducer, grouping_size):
        dmodel = DDP(model, device_ids=None, bucket_cap_mb=grouping_size//1024*1024, find_unused_parameters=True)
        dmodel.sync_gradients = lambda:None
        dmodel.cleanup = lambda:None
        dmodel.parameters = lambda: model.parameters()
        return dmodel

    dummy_reducer = SimpleNamespace(cleanup=lambda:None)
    config.distribute_model = distribute_model
    config.experiment_name = "ddp"

    worker_process(config.rank, 0, config, dummy_reducer)

def main_central_reduce(config):
    config.distribute_model = OurDist
    cpu_reducer = NodeAgreggateReducerCPU(built_in_allreduce, config.node_dev)
    config.experiment_name = "ourdist"

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i, config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()
    worker_process(config.rank, 0, config, dummy_reducer)

def experiment1(config):
    main_ourdist(config)
    main_seq_merge(config)
    main_overlap(config)

def experiment2(config):
    main_ddp(config)
    main_central_reduce(config)
    main_onestep_reduce(config)

def init_process(config):
    """ Initialize the distributed environment. """
    print(config)
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = config.ifname

    dist.init_process_group(config.backend, rank=config.rank, world_size=config.size)

    print("Connection initialised")


if __name__ == "__main__":
    print_d(f"Number of available devices {torch.cuda.device_count()}", Level.INFO)
    config = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    init_process(config)
    experiment1(config)
