# De-centralised Data Parallel Distributed Learning

This project was created for the Cambridge PART III L46 (Principles of Machine Learning Systems course).

## Description
A PyTorch re-implementation of the Horovod distributed learning framework for multi-GPU and multi-node environments.

## Usage
```
main.py [-h] [--job_id JOB_ID] [--experiment EXPERIMENT]
               [--limit_batches LIMIT_BATCHES] [--backend BACKEND]
               n r d D a i M R g

Distributed machine learning benchmark tool

positional arguments:
  n                     number of nodes
  r                     rank (id) of this node
  d                     number of devices managed by this node
  D                     total number of devices
  a                     address of the master node
  i                     network card name to use (eg. lo, ib0, eth0)
  M                     the name of the experiment to run
  R                     root folder for the datasets
  g                     whether to use the gpu or just cpu

optional arguments:
  -h, --help            show this help message and exit
  --job_id JOB_ID       unique string used for results folder name
  --experiment EXPERIMENT
                        which experiment function to run
  --limit_batches LIMIT_BATCHES
                        which experiment function to run
  --backend BACKEND     which experiment function to run
```

Other configuration options are available in `src/config.py`.

There are also two submit files included for the University of Cambridge's HPC:
- `submit.sh`:  Submits a job on the GPU cluster.
- `submit_cpu.sh`: Submits a job on the CPU cluster.
