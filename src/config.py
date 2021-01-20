from utils import eval_arg
import argparse

from network import BasicNet, GoogLeNet
from data import get_partition_mnist, get_partition_image_folder

create_network = { "mnist" : BasicNet, "imagenet" : GoogLeNet}
get_partition_dataset = { "mnist" : get_partition_mnist, "imagenet" : get_partition_image_folder }

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed machine learning benchmark tool')
    parser.add_argument('size', metavar='n', type=str,
                        help='number of nodes')
    parser.add_argument('rank', metavar='r', type=str,
                        help='rank (id) of this node')
    parser.add_argument('node_dev', metavar='d', type=int,
                        help='number of devices managed by this node')
    parser.add_argument('total_dev', metavar='D', type=int,
                        help='total number of devices')
    parser.add_argument('master_addr', metavar='a', type=str, default="127.0.0.1",
                        help='address of the master node')
    parser.add_argument('ifname', metavar='i', type=str, default="lo",
                        help='network card name to use (eg. lo, ib0, eth0)')
    parser.add_argument('model_type', metavar='M', type=str, default="mnist",
                        help='the name of the experiment to run')
    parser.add_argument('dataset_root', metavar='R', type=str, default="../data",
                        help='root folder for the datasets')
    parser.add_argument('use_gpu', metavar='g', type=int, default=0,
                        help='whether to use the gpu or just cpu')
    parser.add_argument('--job_id', dest='job_id', type=str, default="job",
                        help='unique string used for results folder name')
    parser.add_argument('--experiment', dest='experiment', type=str, default="main_ourdist",
                        help='which experiment function to run')
    parser.add_argument('--limit_batches', dest='limit_batches', type=int, default=3,
                        help='which experiment function to run')
    parser.add_argument('--backend', dest='backend', type=str, default="gloo",
                        help='which experiment function to run')
    config = parser.parse_args()
    config.size = int(eval_arg(config.size))
    config.rank = int(eval_arg(config.rank))

    config.create_network = create_network[config.model_type]
    config.get_partition_dataset = get_partition_dataset[config.model_type]

    config.epoch_count = 100
    config.grouping_size = 25*1024*1024

    if config.use_gpu == 1:
        config.devices = [f"cuda:{i}" for i in range(config.node_dev)]
    else:
        config.devices = ["cpu"] * config.node_dev

    return config
