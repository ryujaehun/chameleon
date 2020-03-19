#! /bin/sh
for i in $(squeue |grep jaehunryu|awk '{print $1}')
do
    scancel $i
done        
