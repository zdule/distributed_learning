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

from timing import start_timing_experiment, start_timer, end_timer, writeout_timer
from allreduce import built_in_allreduce, ring_allreduce
from reducers import NodeAgreggateReducerCPU
from ourdist import OurDist, SeqDist, SeqMergeDist
from config import parse_args

def worker_process(node_id, worker_id, config, reducer):
    torch.manual_seed(1234)
    train_set = config.get_partition_dataset(node_id, worker_id, config.node_dev, config.total_dev, config.dataset_root)
    device = config.devices[worker_id]
    model = config.create_network().to(device)
    model = config.distribute_model(model, reducer)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    if node_id == 0 and worker_id == 0:
        loss_f = open("loss.txt", "w")
    else:
        loss_f = None

    start_timing_experiment("run")    
    for epoch in range(config.epoch_count):
        for data, target in islice(train_set,config.limit_batches):
            print_d("WORKER: Moving training data to device", Level.DEBUG)
            start_timer("data2dev")
            data, target = data.to(device), target.to(device)
            end_timer("data2dev")

            optimizer.zero_grad()

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
    
            model.sync_gradients()

            # Optimizer step 
            start_timer("optimizer_step")
            optimizer.step()
            end_timer("optimizer_step")

            if loss_f != None: 
                print(loss)
                loss_f.write(str(loss) + "\n")
    if loss_f != None:
        loss_f.close()
    model.cleanup()
    reducer.cleanup()
    writeout_timer("times.csv")

def worker_process_wrapper(*args):
    try:
        worker_process(*args)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

def main_process(config):
    config.distribute_model = OurDist
    cpu_reducer = NodeAgreggateReducerCPU(built_in_allreduce, config.node_dev)

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i,config, cpu_reducer.reducers[i]))
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

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i,config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_ddp(config):
    def distribute_model(model, reducer):
        dmodel = DDP(model, device_ids=None, bucket_cap_mb=0, find_unused_parameters=True)
        dmodel.sync_gradients = lambda:None
        dmodel.cleanup = lambda:None
        dmodel.parameters = lambda: model.parameters()
        return dmodel

    dummy_reducer = SimpleNamespace(cleanup=lambda:None)
    config.distribute_model = distribute_model

    worker_process(config.rank, 0, config, dummy_reducer)

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
    main_ddp(config)
