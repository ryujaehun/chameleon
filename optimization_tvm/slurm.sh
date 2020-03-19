#!/bin/bash  
#SBATCH  -J nework_vgg-16_batch_4_optlevel_0_tuner_ga_trials_400 
#SBATCH  -o /home/jaehunryu/workspace/tvm/optimization_tvm/results/llvm/2020-03-04_22:48:06.558382/%j.nework_vgg-16_batch_4_optlevel_0_tuner_ga_trials_400.out
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
python3 tune_relay_x86.py --opt_level=0 --n_trial=400 --network=vgg-16 --batch=4 --tuner=ga --time=/home/jaehunryu/workspace/tvm/optimization_tvm/results/llvm/2020-03-04_22:48:06.558382
/home/jaehunryu/linux/tools/perf/perf stat  -d -d -d python3 /home/jaehunryu/workspace/tvm/optimization_tvm/flops.py --path=/home/jaehunryu/workspace/tvm/optimization_tvm/results/llvm/2020-03-04_22:48:06.558382 --batch=4
squeue --job $SLURM_JOBID
