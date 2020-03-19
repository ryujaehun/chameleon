#!/usr/bin/env python
import subprocess,time,datetime,os
from tqdm import tqdm

networks=['vgg-16','resnet-34']
target='llvm '
batchs=['4']
threads=['4']
opts=['0','1','2','3','4']

basetext='''#!/bin/bash  
#SBATCH  -J _name 
#SBATCH  -o inference/%j._name.out
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
/home/jaehunryu/linux/tools/perf/perf stat  -d -d -d python3 /home/jaehunryu/workspace/tvm/optimization_tvm/naive.py --opt_level=_opt  --network=_network --batch=4 

squeue --job $SLURM_JOBID
'''
os.makedirs('/home/jaehunryu/workspace/tvm/optimization_tvm/inference',exist_ok=True)
sript='sbatch slurm_inference.sh'
_list=[]
for network  in networks:
        for opt in opts:
            _list.append([network,opt])
                                 
                        
for idx,pack in enumerate(_list):
    network,opt=pack
    _name='nework_'+network+'_optlevel_'+opt

    text=basetext
    text=text.replace('_opt',opt)
    text=text.replace('_network',network)
    text=text.replace('_node',str(idx%6+1))
    text=text.replace('_name',_name)

    
    num=subprocess.Popen("squeue|grep jaehun|wc -l", shell=True, stdout=subprocess.PIPE).stdout.read()
    num=int(num.decode("utf-8")[:-1])
    while num>20:
        num=subprocess.Popen("squeue|grep jaehun|wc -l", shell=True, stdout=subprocess.PIPE).stdout.read()
        num=int(num.decode("utf-8")[:-1])
        time.sleep(5)
    with open('/home/jaehunryu/workspace/tvm/optimization_tvm/slurm_inference.sh', 'w') as f:
        f.write(text)
    time.sleep(17)
    proc = subprocess.Popen( sript , shell=True, executable='/bin/bash')
    proc.communicate()

