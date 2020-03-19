#!/usr/bin/env python
import subprocess,time,datetime,os
from tqdm import tqdm

networks=['resnet-18','resnet-34','resnet-50','resnet-101','resnet-152','resnet-200','resnet-269','vgg-11','vgg-13','vgg-16','vgg-19','inception_v3','mobilenet','squeezenet_v1.0','squeezenet_v1.1']
target='cuda '
batchs=['1','8','64']
tuners=['xgb','ga','random','gridsearch']
n_trials=['50','500','5000']
basetext='''#!/bin/bash  
#SBATCH  -J _name 
#SBATCH  -o _time/%j._name.out
#SBATCH  -t 1-20:00:00
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1
#SBATCH  --cpus-per-task=2
#SBATCH -p gpu-titanxp

#SBATCH   --gres=gpu:1 
set echo on
cd  $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module  purge
module  load  postech
date
source env/bin/activate
python3 tune_relay_cuda.py  --n_trial=_n_trial --network=_network --batch=_batch --tuner=_tuner --time=_time
/home/jaehunryu/linux/tools/perf/perf stat python3 /home/jaehunryu/workspace/tvm/optimization_tvm/flops_cuda.py --path=_time --batch=_batch
squeue --job $SLURM_JOBID
'''
sript='sbatch slurm.sh'
_list=[]
for network  in networks:
    for batch in batchs:
        for tuner in tuners:
            for trial in n_trials:
                _list.append([network,batch,tuner,trial])
                                 
                        
for idx,pack in enumerate(_list):
    network,batch,tuner,trial=pack
    _name='nework_'+network+'_batch_'+batch+'_tuner_'+tuner+'_trials_'+trial
    _time="_".join(str(datetime.datetime.now()).split())
    _time=os.path.join('/home/jaehunryu/workspace/tvm/optimization_tvm/results/cuda',_time)
    os.makedirs(_time,exist_ok=True)
    text=basetext
    text=text.replace('_name',_name)
    text=text.replace('_n_trial',trial)
    text=text.replace('_network',network)
    text=text.replace('_batch',batch)
    text=text.replace('_tuner',tuner)
    text=text.replace('_time',_time)
    with open('/home/jaehunryu/workspace/tvm/optimization_tvm/slurm.sh', 'w') as f:
        f.write(text)
    proc = subprocess.Popen( sript , shell=True, executable='/bin/bash')
    proc.communicate()
    time.sleep(31)

