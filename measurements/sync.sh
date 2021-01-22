#!/bin/bash
#scp -r cluster:~/repo/results .
#rsync -r cluster:~/repo/results1 gpu1
#rsync -r cluster:~/repo/results gpu2
rsync -r cluster:~/cpu_repo/results cpu
