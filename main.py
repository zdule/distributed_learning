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
        print(i)
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


def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    device = torch.device("cuda:{}".format(0))
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = math.ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(2):
        epoch_loss = 0.0
        for data, target in train_set:

            # Moving training data to cuda device
            data, target = data.to(device), target.to(device)

            # Feed forward and backprop
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()

            # Move gradients to CPU
            grads = move_gradients_to_cpu(model)

            # Ring all-reduce
            for grad in grads:
                ring_all_reduce(grad)

            # Copy back reduced gradients to the GPU
            i = 0
            for param in model.parameters():
                param.grad.data[:] = grads[i]
                i += 1

            # Perform optimiser step
            optimizer.step()

        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '192.168.0.193'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = 'wlo1'

    dist.init_process_group(backend, rank=rank, world_size=size)

    print("Connection initialised")
    fn(rank, size)


if __name__ == "__main__":
    size = int(sys.argv[1])
    rank = int(sys.argv[2])
    p = Process(target=init_process, args=(rank, size, run))

    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        p.kill()
        sys.exit(0)
