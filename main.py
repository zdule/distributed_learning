import os
import math
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.random

from network import Net
from data import partition_dataset

EPOCH_NUM = 2


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
    for param in model.parameters():
        print(param.grad)
        print(param)
    return [param.grad.data.to("cpu") for param in model.parameters()]


def move_gradients_to_gpu_and_optimise(model, grads, optimiser):
    for i, param in enumerate(model.parameters()):
        param.grad.data[:] = grads[i][:]
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
    for param in model.parameters():
        print(param.grad)


def gpu_prcess(device, train_set, to_cpu_queue, from_cpu_queue):
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for _epoch in range(EPOCH_NUM):

        for data, target in train_set:

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Feed forward
            output = model(data)
            loss = F.nll_loss(output, target)

            # Backpropagation
            loss.backward()

            # Share result to CPU
            for param in model.parameters:
                to_cpu_queue.put(param.grad.data)

            # Get all recuced gradient
            for param in model.parameters:
                param.grad.data[:] = from_cpu_queue.get()

            # Send back loss for this batch (this also signals
            # that everything has been received)
            to_cpu_queue.put(loss.item())

            optimizer.step()


def main_process(rank, size, node_dev, total_dev):
    torch.manual_seed(1234)
    train_sets, bsz = partition_dataset(node_dev, total_dev)
    num_batches = len(train_sets[0])

    devices = [torch.device("cuda:{}".format(i)) for i in range(node_dev)]
    to_cpu_queues = [mp.Queue() for _ in devices]
    from_cpu_queues = [mp.Queue() for _ in devices]

    buffer_model = Net()

    for args in zip(devices, train_sets, to_cpu_queues, from_cpu_queues):
        p = mp.Process(target=gpu_prcess, args=args)
        p.start()

    for epoch in range(EPOCH_NUM):
        epoch_losses = [0.0 for _ in devices]

        for b in range(num_batches):

            # Reset the collected parameters
            for param in buffer_model.parameters():
                param[:] = 0

            # Get the local parameters and allreduce layer-by-layer
            for param in buffer_model.parameters():
                for queue in to_cpu_queues:
                    received = queue.get()
                    param[:] += received[:]
                    del received
                ring_all_reduce(param)
                for queue in from_cpu_queues:
                    queue.put(param)

            # Grab the loss from workers
            for queue, epoch_loss in zip(to_cpu_queues, epoch_losses):
                epoch_loss += queue.get()

        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', sum(epoch_losses) / num_batches / node_dev)


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
    p = mp.Process(target=init_process, args=(
        rank, size, node_dev, total_dev, main_process))

    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        p.kill()
        sys.exit(0)
