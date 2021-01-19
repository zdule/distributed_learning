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
import datetime

from itertools import islice
from types import SimpleNamespace

from utils import Level, print_d, eval_arg

from timing import end_timing_experiment, start_timer, end_timer, writeout_timer
import time
from allreduce import built_in_allreduce, ring_allreduce, ring_allreduce_gpu, central_allreduce
from reducers import NodeAgreggateReducerCPU, ReduceImmediatelly
from ourdist import OurDist, SeqDist, SeqMergeDist, WarmupDist
from config import parse_args

pgroups = []

def worker_process(node_id, worker_id, config, reducer, gpu_reduce=False):
    print_d(f"Starting experiment {config.experiment}, {datetime.datetime.now()}", Level.INFO)
    torch.manual_seed(1234)
    train_set = config.get_partition_dataset(node_id, worker_id, config.node_dev, config.total_dev, config.dataset_root)
    device = config.devices[worker_id]
    model = config.create_network().to(device)
    model = config.distribute_model(model, reducer, config.grouping_size, "cpu" if not gpu_reduce else device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
   
    config_f = open(f"{config.folder}/{config.experiment_name}_config.txt", "w") 
    config_f.write(str(config))
    config_f.close()
    loss_f = open(f"{config.folder}/{config.experiment_name}_{node_id}_{worker_id}_loss.txt", "w")
    
    start_time = time.time()
    batch_count = 0
    for epoch in range(config.epoch_count):
        if batch_count >= config.limit_batches:
            break
        gener = train_set.__iter__()
        while True:
            if batch_count >= config.limit_batches:
                break
            start_timer("batch")

            start_timer("get_data")
            nx = next(gener, None)
            if nx == None: break
            data, target = nx
            end_timer("get_data")


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
            end_timing_experiment(config.experiment_name, extra_fields={"batch_count" : batch_count, "data_len" : data.size(0)})
            batch_count += 1

    end_time = time.time()

    loss_f.close()
    model.cleanup()
    reducer.cleanup()

    writeout_timer(f"{config.folder}/{config.experiment_name}_{node_id}_{worker_id}_times.csv")

def worker_process_wrapper(*args):
    try:
        worker_process(*args)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

def main_warmup(config):
    config.distribute_model = WarmupDist
    dummy_reducer = SimpleNamespace(cleanup=print)
    config.experiment_name = "warmup"

    procs = []
    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i, config, dummy_reducer))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

def main_ourdist(config):
    config.distribute_model = OurDist
    cpu_reducer = NodeAgreggateReducerCPU(ring_allreduce, config.node_dev)
    config.experiment_name = "ourdist"

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i, config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_ourdist_nccl(config):
    config.distribute_model = OurDist
    cpu_reducer = NodeAgreggateReducerCPU(ring_allreduce_gpu, config.node_dev, pgroups)
    config.experiment_name = "ourdist_nccl"

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i, config, cpu_reducer.reducers[i], True))
        p.start()
    cpu_reducer.pump()

def main_seq(config):
    config.distribute_model = SeqDist
    cpu_reducer = NodeAgreggateReducerCPU(ring_allreduce, config.node_dev)

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i,config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_seq_merge(config):
    config.distribute_model = SeqMergeDist
    cpu_reducer = NodeAgreggateReducerCPU(ring_allreduce, config.node_dev)
    config.experiment_name = "seq_merge"

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i,config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_overlap(config):
    config.distribute_model = OurDist
    cpu_reducer = NodeAgreggateReducerCPU(ring_allreduce, config.node_dev)
    config.experiment_name = "overlap"
    old_grouping = config.grouping_size
    config.grouping_size = 0

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i, config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()
    config.grouping_size = old_grouping

def main_ddp(config):
    def distribute_model(model, reducer, grouping_size, _grad_buffer_device="cpu"):
        device_ids = None
        if config.devices[0] != "cpu":
            device_ids = [config.device_id]
        dmodel = DDP(model, device_ids=device_ids, bucket_cap_mb=grouping_size/1024/1024, find_unused_parameters=True)
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
    cpu_reducer = NodeAgreggateReducerCPU(central_allreduce, config.node_dev)
    config.experiment_name = "central_node_reduce"

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i, config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_onestep_reduce(config):
    config.distribute_model = OurDist
    cpu_reducer = ReduceImmediatelly(ring_allreduce)
    config.experiment_name = "onestep_reduce"

    worker_process(config.rank, 0, config, cpu_reducer)

def main_single(config):
    def distribute_model(model, reducer, grouping_size, _grad_buffer_device="cpu"):
        model.sync_gradients = lambda:None
        model.cleanup = lambda:None
        return model
    config.distribute_model = distribute_model
    dummy_reducer = SimpleNamespace(cleanup=lambda:None)
    config.experiment_name = "single"

    worker_process(0, 0, config, dummy_reducer)

def experiment1(config):
    main_warmup(config)
    main_ourdist(config)
    main_seq_merge(config)
    main_overlap(config)
    main_central_reduce(config)

def experiment2(config):
    main_warmup(config)
    main_ddp(config)
    main_onestep_reduce(config)

def experiment_nccl(config):
    main_warmup(config)
    main_ddp(config)
    main_ourdist_nccl(config)

fusion_test_sizes_k = [1024, 4*1024, 16*1024, 64*1024]
def fusion_experiment(config, main_f):
    main_warmup(config)
    folder_name = config.folder
    for size_k in fusion_test_sizes_k:
        config.grouping_size = size_k*1024
        config.folder = folder_name + f"/{size_k}"
        os.makedirs(config.folder, exist_ok=True)
        main_f(config)

def fusion_experiment_ddp(config):
    fusion_experiment(config, main_ddp)

def fusion_experiment_ourdist(config):
    fusion_experiment(config, main_ourdist)

def init_process(config):
    """ Initialize the distributed environment. """
    print(config)
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = config.ifname

    dist.init_process_group(config.backend, rank=config.rank, world_size=config.size)
    
    global pgroups 
    pgroups = []
    if config.size > 1:
        for i in range(config.size):
            pgroups.append(dist.new_group([i,(i+1)%config.size]))

    print("Connection initialised")


if __name__ == "__main__":
    print_d(f"Number of available devices {torch.cuda.device_count()}", Level.INFO)
    config = parse_args()
    config.folder = f"results/{config.experiment}_{config.total_dev}_{config.job_id}"
    os.makedirs(config.folder, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')
    init_process(config)

    if config.node_dev == 1 and config.devices[0].startswith("cuda"):
        config.device_id = config.rank%torch.cuda.device_count()
        config.devices = [f"cuda:{config.device_id}" for _ in config.devices]
    #experiment1(config)
    globals()[config.experiment](config)
