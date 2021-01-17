import os
import math
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.random

from network import BasicNet, GoogLeNet
from data import partition_mnist, partition_image_net, partition_image_folder
from utils import Level, print_d, eval_arg
import traceback
import threading

from timing import start_timing_experiment, start_timer, end_timer, writeout_timer

EPOCH_NUM = 1
sync_method = "fused"

create_network = { "mnist" : BasicNet, "imagenet" : GoogLeNet }
partition_dataset = { "mnist" : partition_mnist, "imagenet" : partition_image_folder }
process_output = { "mnist" : lambda x : x, "imagenet" : lambda x : F.log_softmax(x[0], dim=1) }

def built_in_allreduce(send):
    dist.all_reduce(send, op=dist.reduce_op.SUM)
    send /= float(dist.get_world_size())


def ring_all_reduce(send):
    """
    Custom ring all-reduce fucntion that averages the gradients.

    :param send: The gradients computed at an individual node. The
        function overwrites them with the shared averages.
    """

    rank = dist.get_rank()
    size = dist.get_world_size()
    if size <= 1:
        return

    chunks = torch.chunk(send, size)
    maxsize = max((chunk.size() for chunk in chunks))
    recv_buffer = torch.empty(maxsize, dtype=send.dtype)

    right = (rank + 1) % size
    left = (rank-1) % size

    send_reqs = []

    # First passDEBUG_LEVEL
    for i in range(size - 1):
        to_send = (rank-i) % size
        to_recv = (to_send - 1) % size
        send_reqs.append(dist.isend(chunks[to_send], right))
        dist.recv(recv_buffer, left)        # Receiving needs to be blocking
        chunks[to_recv][:] += recv_buffer[:len(chunks[to_recv])]

    for send_req in send_reqs:
        send_req.wait()     # Need to wait till sending is finished
    send_reqs = []

    # We now have result[r+1] on node with rank r

    # Second pass
    for i in range(size-1):
        to_send = (rank - i + 1) % size
        to_recv = (to_send - 1) % size
        send_reqs.append(dist.isend(chunks[to_send], right))
        dist.recv(recv_buffer, left)         # Receiving needs to be blocking
        chunks[to_recv][:] = recv_buffer[:len(
            chunks[to_recv])]  # [:] indicates deepcopy

    for send_req in send_reqs:
        send_req.wait()     # Need to wait till sending is finished

    # Dividing result by the number of devices
    send /= float(size)

"""
def move_gradients_to_cpu(model):
    for param in model.parameters():
        print(param.grad)
        print(param)
    return [param.grad.data.to("cpu") for param in model.parameters()]


def move_gradients_to_gpu_and_optimise(model, grads, optimiser):
    for i, param in enumerate(model.parameters()):
        param.grad.data[:] = grads[i][:]
    optimiser.step()
"""

def fusion_grouping_gen(model, do_print=False):
    params = model.parameters()
    finished = False
    while not finished:
        to_fuse = []
        total_size = 0
        while total_size < 40000:
            nxt = next(params, None)
            if nxt == None:
                finished = True
                break
            if nxt.requires_grad == False or nxt.grad == None:
                continue
            nxt = nxt.grad.data
            to_fuse.append(nxt)
            total_size += nxt.numel()
        if len(to_fuse) > 0:
            yield to_fuse

def load_fused_params_from_cpu(from_cpu_queue, model, device):
    for to_fuse in fusion_grouping_gen(model):
        result = from_cpu_queue.get()
        curr = 0
        for t in to_fuse:
            t.view(-1)[:] = result[curr:curr+t.numel()]
            curr += t.numel()
        del result

def load_params_from_cpu(from_cpu_queue, model, device):
    for param in model.parameters():
        if param.requires_grad == False or param.grad == None:
            continue
        result = from_cpu_queue.get()
        result_dev = result.to(device)
        del result
        param.grad.data[:] = result_dev

def fuse_to_cpu(tensors):
    if len(tensors) == 0: return None
    required_size = sum((t.numel() for t in tensors))
    result = torch.empty(required_size, dtype=tensors[0].dtype)
    curr = 0
    for t in tensors:
        result[curr:curr+t.numel()] = t.view(-1)[:]
        curr += t.numel()
    return result


def gpu_process(device, train_set, to_cpu_queue, from_cpu_queue, network_type):
    model = create_network[network_type]().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    worker_loss = 0

    start_timing_experiment("Run")
    for epoch in range(EPOCH_NUM):
        print_d(f"GPU: Starting epoch {epoch}", Level.INFO)

        itr = 0
        for data, target in train_set:
            sys.stderr.flush()
            itr += 1
            if itr > 2: break

            print_d("GPU: Moving training data to GPU", Level.DEBUG)
            start_timer("data2gpu")
            data, target = data.to(device), target.to(device)
            end_timer("data2gpu")
            optimizer.zero_grad()

            # Feed forward
            print_d("GPU: Perfroming feed forward and backprop", Level.DEBUG)
            start_timer("compute")
            output = model(data)
            loss = F.nll_loss(process_output[network_type](output), target)
            end_timer("compute")

            start_timer("backprop")
            # Backpropagation
            loss.backward()
            end_timer("backprop")

            # Share result to CPU and read back
            print_d("GPU: Communicating with CPU", Level.DEBUG)
            start_timer("sync")
            if sync_method == "sync":
                for param in model.parameters():
                    if param.requires_grad == False or param.grad == None:
                        continue
                    print_d("GPU: Sending local grad", Level.DEBUG)
                    to_cpu_queue.put(param.grad.data)

                    print_d("GPU: Receving global grad", Level.DEBUG)
                    result = from_cpu_queue.get()
                    result_dev = result.to(device)
                    del result
                    param.grad.data[:] = result_dev
            elif sync_method == "async":
                print("async")
                t = threading.Thread(target=load_params_from_cpu, args=(from_cpu_queue, model, device))
                t.start()
                for param in model.parameters():
                    if param.requires_grad == False or param.grad == None:
                        continue
                    print_d("GPU: Sending local grad", Level.DEBUG)
                    to_cpu_queue.put(param.grad.data.to("cpu"))
                t.join()
            else:
                print("fused")
                t = threading.Thread(target=load_fused_params_from_cpu, args=(from_cpu_queue, model, device))
                t.start()
                params = model.parameters()
                finished = False
                for to_fuse in fusion_grouping_gen(model):
                    fused = fuse_to_cpu(to_fuse)
                    to_cpu_queue.put(fused)
                t.join()

            # Send back loss for this batch (this also signals
            # that everything has been received)

            print_d("GPU: Sending loss to CPU", Level.DEBUG)
            worker_loss = loss.item()
            to_cpu_queue.put(worker_loss)
            end_timer("sync")
 
            start_timer("optimizer_step")
            optimizer.step()
            end_timer("optimizer_step")
    writeout_timer("times.csv")


def gpu_process_wrapper(*args):
    try:
        gpu_process(*args)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

def prime_model(model, sample, network_type):
    print_d("Priming CPU model", Level.DEBUG)
    data, target = sample
    output = model(data)
    loss = F.nll_loss(process_output[network_type](output), target)
    loss.backward()
    print_d("Done priming CPU model", Level.DEBUG)

def main_process(rank, size, node_dev, total_dev, network_type, dataset_root):
    torch.manual_seed(1234)
    train_sets, bsz = partition_dataset[network_type](node_dev, total_dev, dataset_root)
    num_batches = len(train_sets[0])

    #devices = [torch.device("cuda:{}".format(i)) for i in range(node_dev)]
    devices = [torch.device("cpu") for i in range(node_dev)]
    to_cpu_queues = [mp.Queue(maxsize=50) for _ in devices]
    from_cpu_queues = [mp.Queue() for _ in devices]

    buffer_model = create_network[network_type]()
    prime_model(buffer_model, train_sets[0].__iter__().next(), network_type)

    for args in zip(devices, train_sets, to_cpu_queues, from_cpu_queues, [network_type] * node_dev):
        p = mp.Process(target=gpu_process_wrapper, args=args)
        p.start()

    for epoch in range(EPOCH_NUM):
        epoch_losses = [0.0 for _ in devices]
        print_d(f"CPU: Starting epoch {epoch}", Level.INFO)

        itr = 0
        for b in range(num_batches):
            itr += 1
            if itr > 2: break

            print_d(f"CPU: Starting batch {b}", Level.DEBUG)
            # Reset the collected parameters
            for param in buffer_model.parameters():
                if param.requires_grad == False or param.grad == None:
                    continue
                param.grad.data[:] = 0
                param.data[:] = 0

 
            print_d(f"CPU: Receiving local grad from GPUs", Level.DEBUG)
            if sync_method == "sync" or sync_method == "async":
                print_d("Not Fused", Level.DEBUG)
                for param in buffer_model.parameters():
                    for que in to_cpu_queues:
                        received = que.get()
                        received_cpu = received.to("cpu")
                        del received
                        param[:] += received_cpu[:]
                    param[:] /= float(node_dev)

                    print_d(f"CPU: Performing allreduce", Level.DEBUG)
                    ring_all_reduce(param)
                    print_d(f"CPU: Sending global grad to GPUs", Level.DEBUG)
                    for que in from_cpu_queues:
                        que.put(param.detach())
            else:
                for to_fuse in fusion_grouping_gen(buffer_model, do_print=True):
                    print_d(f"CPU: Receiving local grad from GPUs", Level.DEBUG)
                    param = torch.zeros(sum((t.numel() for t in to_fuse)), dtype = to_fuse[0].dtype)
                    for que in to_cpu_queues:
                        received = que.get()
                        param[:] += received
                    param[:] /= float(node_dev)
                    print_d(f"Getting tensor {param.numel()}", Level.INFO)

                    print_d(f"CPU: Performing allreduce", Level.DEBUG)
                    ring_all_reduce(param)
                    print_d(f"CPU: Sending global grad to GPUs", Level.DEBUG)
                    for que in from_cpu_queues:
                        que.put(param.detach())

            # Grab the loss from workers
            print_d(f"CPU: Receiving loss from GPUs", Level.DEBUG)
            for i, que in enumerate(to_cpu_queues):
                rec = que.get()
                epoch_losses[i] += rec
                print(epoch_losses[i])

        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', sum(epoch_losses) / num_batches / node_dev)


def main_process_wrapper(*args):
    try:
        main_process(*args)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()


def init_process(rank, size, node_dev, total_dev, master_addr, ifname, network_type, dataset_root, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = ifname

    dist.init_process_group(backend, rank=rank, world_size=size)

    print("Connection initialised")
    fn(rank, size, node_dev, total_dev, network_type, dataset_root)


if __name__ == "__main__":
    size = int(eval_arg(sys.argv[1]))
    rank = int(eval_arg(sys.argv[2]))
    node_dev = int(eval_arg(sys.argv[3]))
    total_dev = int(eval_arg(sys.argv[4]))
    master_addr = eval_arg(sys.argv[5])
    ifname = eval_arg(sys.argv[6])
    network_type = eval_arg(sys.argv[7])
    if len(sys.argv) < 9:
        dataset_root = "../data"
    else:
        dataset_root = eval_arg(sys.argv[8])

    torch.multiprocessing.set_start_method('spawn')
    p = mp.Process(target=init_process, args=(
        rank, size, node_dev, total_dev, master_addr, ifname, network_type, dataset_root, main_process_wrapper))

    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        p.kill()
        sys.exit(0)
