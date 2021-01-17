import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.random

import os, sys
import traceback
import threading

from itertools import islice

from utils import Level, print_d, eval_arg

from timing import start_timing_experiment, start_timer, end_timer, writeout_timer
from allreduce import built_in_allreduce, ring_allreduce
from reducers import NodeAgreggateReducerCPU
from ourdist import OurDist
from config import parse_args

def worker_process(node_id, worker_id, config, reducer):
    torch.manual_seed(1234)
    train_set = config.get_partition_dataset(node_id, worker_id, config.node_dev, config.total_dev, config.dataset_root)
    device = config.devices[worker_id]
    model = config.create_network().to(device)
    model = config.distribute_model(model, reducer)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

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
    cpu_reducer = NodeAgreggateReducerCPU(built_in_allreduce, config.node_dev)

    for i in range(config.node_dev):
        p = mp.Process(target=worker_process_wrapper, args=(config.rank, i,config, cpu_reducer.reducers[i]))
        p.start()
    cpu_reducer.pump()

def main_process_wrapper(*args):
    try:
        main_process(*args)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()


def init_process(config, fn):
    """ Initialize the distributed environment. """
    print(config)
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = config.ifname

    dist.init_process_group(config.backend, rank=config.rank, world_size=config.size)

    print("Connection initialised")
    fn(config)


if __name__ == "__main__":
    config = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    p = mp.Process(target=init_process, args=(config, main_process_wrapper))

    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        p.kill()
        sys.exit(0)
