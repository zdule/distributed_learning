import torch
import torch.distributed as dist
from utils import Level, print_d, eval_arg

from timing import start_timing_experiment, start_timer, end_timer, writeout_timer
from allreduce import built_in_allreduce, ring_allreduce, ring_allreduce_gpu
from config import parse_args
import os

def benchmark_reduce(node_id, worker_id, config, reduce_fn):
    a = torch.empty(60*1024*1024//4, dtype=torch.float32, device=config.devices[worker_id])
    
    reduce_fn(a)
    for i in range(10):
        start_timer("sync"+str(reduce_fn))
        reduce_fn(a)
        end_timer("sync"+str(reduce_fn))

on_cpu = None
def move_to_cpu_reduce(send):
    if on_cpu == None:
        on_cpu = send.to("cpu")
    else:
        on_cpu[:] = send
    ring_allreduce(on_cpu)
    send[:] = on_cpu
    
def benchmarker(config) :
    start_timing_experiment("run")    
    benchmark_reduce(config.rank, 0, config, lambda x: ring_allreduce_gpu(x, config.groups))
    benchmark_reduce(config.rank, 0, config, move_to_cpu_reduce)
    benchmark_reduce(config.rank, 0, config, built_in_allreduce)
    writeout_timer("reduce_timings.csv")

def init_process(config):
    """ Initialize the distributed environment. """
    print(config)
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = config.ifname

    dist.init_process_group(config.backend, rank=config.rank, world_size=config.size)

    if config.groups == []:
        for i in range(config.size):
            config.groups.append(dist.new_group([i,(i+1)%config.size]))

    print("Connection initialised")


if __name__ == "__main__":
    print_d(f"Number of available devices {torch.cuda.device_count()}", Level.INFO)
    config = parse_args()
    config.groups = []
    config.backend = "gloo"
    torch.multiprocessing.set_start_method('spawn')
    init_process(config)
    benchmarker(config)
