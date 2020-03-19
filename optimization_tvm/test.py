#!/usr/bin/env python
import subprocess,time,datetime,os
from tqdm import tqdm

networks=['vgg-16','resnet-34']
target='llvm '
batchs=['4']
opts=['0','1','2','3','4']
tuners=['xgb','ga','random','gridsearch']
n_trials=['400','2000']

basetext='''#!/bin/bash  
#SBATCH  -J _name 
#SBATCH  -o _time/%j._name.out
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
python3 tune_relay_x86.py --opt_level=_opt --n_trial=_n_trial --network=_network --batch=_batch --tuner=_tuner --time=_time
/home/jaehunryu/linux/tools/perf/perf stat  -d -d -d python3 /home/jaehunryu/workspace/tvm/optimization_tvm/flops.py --path=_time --batch=_batch
squeue --job $SLURM_JOBID
'''
sript='sbatch slurm.sh'
_list=[]
for network  in networks:
    for batch in batchs:
        for opt in opts:
            for tuner in tuners:
                for trial in n_trials:
                    _list.append([network,batch,opt,tuner,trial])
                                 
                        
for idx,pack in enumerate(_list):
    network,batch,opt,tuner,trial=pack
    _name='nework_'+network+'_batch_'+batch+'_optlevel_'+opt+'_tuner_'+tuner+'_trials_'+trial
    _time="_".join(str(datetime.datetime.now()).split())
    _time=os.path.join('/home/jaehunryu/workspace/tvm/optimization_tvm/results/llvm',_time)
    os.makedirs(_time,exist_ok=True)
    text=basetext
    text=text.replace('_opt',opt)
    text=text.replace('_n_trial',trial)
    text=text.replace('_network',network)
    text=text.replace('_batch',batch)
    text=text.replace('_tuner',tuner)
    text=text.replace('_time',_time)
    text=text.replace('_name',_name)
    
    num=subprocess.Popen("squeue|grep jaehun|wc -l", shell=True, stdout=subprocess.PIPE).stdout.read()
    num=int(num.decode("utf-8")[:-1])
    while num>20:
        num=subprocess.Popen("squeue|grep jaehun|wc -l", shell=True, stdout=subprocess.PIPE).stdout.read()
        num=int(num.decode("utf-8")[:-1])
        time.sleep(10)
    with open('/home/jaehunryu/workspace/tvm/optimization_tvm/slurm.sh', 'w') as f:
        f.write(text)
    time.sleep(31)
    proc = subprocess.Popen( sript , shell=True, executable='/bin/bash')
    proc.communicate()

