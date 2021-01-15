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


def _partition_helper(node_dev, total_dev, dataset):
    """ 
    Partitioning the given dataset. We assume each node has the same number of devices.
    """

    bsz = PER_WORKER_BATCH_SIZE
    partition_sizes = [1.0 / total_dev for _ in range(total_dev)]
    partition = DataPartitioner(dataset, partition_sizes)
    partitions = [partition.use(dist.get_rank() * node_dev + i)
                  for i in range(node_dev)]
    train_sets = [torch.utils.data.DataLoader(partition,
                                              batch_size=int(bsz),
                                              shuffle=True)
                  for partition in partitions]
    return train_sets, bsz


def partition_mnist(node_dev, total_dev):
    """ 
    Loads and partitions the MNIST dataset
    """

    dataset = datasets.MNIST('../data/mnist', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    return _partition_helper(node_dev, total_dev, dataset)


def partition_image_net(node_dev, total_dev):
    """ 
    Loads and partitions the MNIST dataset
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageNet('../data/ImageNet',
                                download=[
                                    'airplane', 'automobile', 'bird', 'cat',
                                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                ]))

    return _partition_helper(node_dev, total_dev, dataset)

def partition_image_folder(node_dev, total_dev):
    """ 
    Loads and partitions the ImageFolder dataset
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder('data/ImageFolder',
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                ]))

    return _partition_helper(node_dev, total_dev, dataset)
