#!/bin/bash  
#SBATCH  -J nework_vgg-16_optlevel_1 
#SBATCH  -o inference/%j.nework_vgg-16_optlevel_1.out
#SBATCH  -t 1-20:00:00
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1
#SBATCH  --cpus-per-task=4
#SBATCH -p cpu-xg6230
#SBATCH  --nodelist=n14

set echo on
cd  $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module  purge
module  load  postech
date
source env/bin/activate
/home/jaehunryu/linux/tools/perf/perf stat  -d -d -d python3 /home/jaehunryu/workspace/tvm/optimization_tvm/naive.py --opt_level=1  --network=vgg-16 --batch=4 

squeue --job $SLURM_JOBID
