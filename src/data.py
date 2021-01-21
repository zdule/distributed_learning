import torch
import torch.distributed as dist
import torch.random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

PER_WORKER_BATCH_SIZE = 128


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

    def __init__(self, data, total_dev, batch_size, seed=1234):
        self.data = data
        self.partitions = []
        data_len = len(data)
        batches_per_dev = data_len // total_dev // batch_size
        part_len = batches_per_dev * batch_size
        indexes = torch.randperm(data_len)

        for _ in range(total_dev):
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def _partition_helper(node_dev, total_dev, dataset, batch_size=PER_WORKER_BATCH_SIZE):
    """ 
    Partitioning the given dataset. We assume each node has the same number of devices.
    """

    partition = DataPartitioner(dataset, total_dev, batch_size)
    partitions = [partition.use(dist.get_rank() * node_dev + i)
                  for i in range(node_dev)]
    train_sets = [torch.utils.data.DataLoader(partition,
                                              batch_size=int(batch_size),
                                              shuffle=True)
                  for partition in partitions]
    return train_sets, batch_size

def _get_partition_helper(node_id, worker_id, node_dev, total_dev, dataset, batch_size=PER_WORKER_BATCH_SIZE):
    """ 
    Get one partition from a given dataset.
    """

    partitioner = DataPartitioner(dataset, total_dev, batch_size)
    partition = partitioner.use(node_id * node_dev + worker_id)
    train_set = torch.utils.data.DataLoader(partition,
                                              batch_size=int(batch_size),
                                              shuffle=True, num_workers = 2)
    return train_set


def partition_mnist(node_dev, total_dev, dataset, batch_size=PER_WORKER_BATCH_SIZE):
    """ 
    Loads and partitions the MNIST dataset
    """

    dataset = datasets.MNIST(dataset + '/mnist', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    return _partition_helper(node_dev, total_dev, dataset, batch_size)

def get_partition_mnist(node_id, worker_id, node_dev, total_dev, dataset, batch_size=PER_WORKER_BATCH_SIZE):
    """ 
    Loads and partitions the MNIST dataset
    """

    dataset = datasets.MNIST(dataset + '/mnist', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    return _get_partition_helper(node_id, worker_id, node_dev, total_dev, dataset, batch_size)

def partition_image_folder(node_dev, total_dev, dataset_root, batch_size=PER_WORKER_BATCH_SIZE):
    """ 
    Loads one partition of the ImageFolder dataset
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(dataset_root + '/ImageFolder',
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                ]))

    return _partition_helper(node_dev, total_dev, dataset, batch_size)

def get_partition_image_folder(node_id, worker_id, node_dev, total_dev, dataset_root, batch_size=PER_WORKER_BATCH_SIZE):
    """ 
    Loads one partition of the ImageFolder dataset
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(dataset_root + '/ImageFolder',
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                ]))

    return _get_partition_helper(node_id, worker_id, node_dev, total_dev, dataset, batch_size)

def random_data_generator(data_pair):
    data, target = data_pair
    while True:
        yield torch.rand_like(data), torch.randint_like(target,high=target.max()+1)
