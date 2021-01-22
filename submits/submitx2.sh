#!/bin/bash

#! Name of the job:
#SBATCH -J ex1x2
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A LANE-SL3-GPU 
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 3 cpus per GPU.
#SBATCH --gres=gpu:2
#! How much wallclock time will be required?
#SBATCH --time=00:20:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue
#! Do not change:
#SBATCH -p pascal

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

# leaving this out might limit the number of threads available
unset OMP_NUM_THREADS

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment
. ~/testenv/bin/activate
#! Insert additional module load commands after this line if needed:

ip_addr=`ip addr show ib0 | grep -Po 'inet \K[\d.]+'`
echo $ip_addr
total_gpus=$[${numtasks}*4]
echo $total_gpus

#! Full path to application executable: 
application="python3 src/main.py"

#! Run options for the application:
options="envarg://OMPI_COMM_WORLD_SIZE envarg://OMPI_COMM_WORLD_RANK 2 2 $ip_addr ib0 imagenet /rds/user/dz308/hpc-work 1 --job_id $SLURM_JOB_ID --experiment experiment1 --limit_batches 30 --random_input 1"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
#CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
