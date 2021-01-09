import os
import math
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.random
from torch.multiprocessing import Process

from network import Net
from data import partition_dataset


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
    chunks = torch.chunk(send, size)
    recv_buffer = chunks[0].clone()

    right = (rank + 1) % size
    left = (rank-1) % size

    # First pass
    for i in range(size - 1):
        to_send = (rank-i) % size
        to_recv = (to_send - 1) % size
        send_req = dist.isend(chunks[to_send], right)
        dist.recv(recv_buffer, left)        # Receiving needs to be blocking
        chunks[to_recv][:] += recv_buffer[:]

    send_req.wait()     # Need to wait till sending is finished

    # We now have result[r+1] on node with rank r

    # Second pass
    for i in range(size-1):
        to_send = (rank - i + 1) % size
        to_recv = (to_send - 1) % size
        send_req = dist.isend(chunks[to_send], right)
        dist.recv(recv_buffer, left)         # Receiving needs to be blocking
        chunks[to_recv][:] = recv_buffer[:]  # [:] indicates deepcopy

    send_req.wait()     # Need to wait till sending is finished

    # Dividing result by the number of devices
    send /= float(size)


def move_gradients_to_cpu(model):
    return [param.grad.data.to("cpu") for param in model.parameters()]


def move_gradients_to_gpu_and_optimise(model, grads, optimiser):
    for i, param in enumerate(model.parameters()):
        param.grad.data[:] = grads[i]
    optimiser.step()


def forward_and_backprop(device, model, optimizer, data, target, loss_ret):
    # Moving training data to cuda device
    data, target = data.to(device), target.to(device)

    # Feed forward and backprop
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss_ret.value = loss.item()
    loss.backward()


def run(rank, size, node_dev, total_dev):
    torch.manual_seed(1234)
    train_sets, bsz = partition_dataset(node_dev, total_dev)
    train_iters = [iter(train_set) for train_set in train_sets]

    devices = [torch.device("cuda:{}".format(i)) for i in range(node_dev)]
    models = [Net().to(device) for device in devices]
    optimizers = [optim.SGD(model.parameters(),
                            lr=0.01, momentum=0.5) for model in models]

    num_batches = math.ceil(len(train_sets[0].dataset) / float(bsz))
    for epoch in range(2):
        epoch_loss = 0.0

        for b in range(num_batches):

            # Run feed-forward and backprop for each model
            processes = []
            for i in range(node_dev):
                data, target = next(train_iters[i])
                loss = torch.multiprocessing.Value("d", 0.0, lock=False)
                p = Process(
                    target=forward_and_backprop,
                    args=(
                        devices[i],
                        models[i],
                        optimizers[i],
                        data,
                        target,
                        loss
                    )
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            # Summing local gradients
            grads = [move_gradients_to_cpu(model) for model in models]
            for g, grad in enumerate(grads[0]):
                for dev in range(1, node_dev):
                    grad[:] += grads[dev][g]
            local_grad = grads[0]

            # All-reduce grads across multiple nodes
            ring_all_reduce(local_grad)

            # Move results back to GPU and perform optimising
            processes = []
            for i in range(node_dev):
                p = Process(
                    target=move_gradients_to_gpu_and_optimise,
                    args=(models[i], local_grad, optimizers[i])
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)


def init_process(rank, size, node_dev, total_dev, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '192.168.0.193'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = 'wlo1'

    dist.init_process_group(backend, rank=rank, world_size=size)

    print("Connection initialised")
    fn(rank, size, node_dev, total_dev)


if __name__ == "__main__":
    size = int(sys.argv[1])
    rank = int(sys.argv[2])
    node_dev = int(sys.argv[3])
    total_dev = int(sys.argv[4])

    torch.multiprocessing.set_start_method('spawn')
    p = Process(target=init_process, args=(
        rank, size, node_dev, total_dev, run))

    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        p.kill()
        sys.exit(0)
