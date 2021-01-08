import os
import math
import sys
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.multiprocessing import Process


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Partition(object):

    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = torch.randperm(data_len)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset():
    """ Partitioning MNIST """

    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)         # Divide the batch by the no. workers
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=int(bsz),
                                            shuffle=True)
    return train_set, bsz


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
        chunks[to_recv] += recv_buffer

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


# def average_gradients():
#     for param in model.parameters():
#         ring_all_reduce(param.grad.data)

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
    # os.environ['WORLD_SIZE'] = '2'
    # os.environ['RANK'] = '0'
    # os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['GLOO_SOCKET_IFNAME'] = 'wlo1'
    # print(os.environ.get('GLOO_SOCKET_IFNAME'))
    os.environ['TP_SOCKET_IFNAME'] = 'wlo1'
    dist.init_process_group(backend,
                            # init_method="tcp://192.168.0.193:29501",
                            rank=rank, world_size=size)

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
