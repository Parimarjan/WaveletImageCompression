#!/bin/bash -l 
#PBS -l nodes=4:ppn=24
#PBS -l walltime=00:05:00
#PBS -m abe
#PBS -q gpu
#PBS -d .

rm ./*.png
rm ./run.sh.*
rm ./logs/*.log


LAUNCHER='mpirun --bind-to none -np 4 -npernode 1' regent.py ~/wic/wavelet.rg -i \
~/wic/images/gates.png -ll:cpu 8 -ll:csize 16000 \
-hl:prof 4 -logfile ~/wic/run/logs/wavelet_%.log

# Now we can generate legion prof files:
#~/legion/tools/legion_prof.py wavelet_0.log

# This will be with profiling + spy
#LAUNCHER='mpirun --bind-to none -np 2 -npernode 1' regent.py ~/wic/wavelet.rg -i \
#~/wic/images/gates.png -ll:cpu 4 -ll:csize 1024 -hl:prof 1 \
#-hl:spy -logfile wavelet_%.log
 

