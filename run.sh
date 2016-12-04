#!/bin/bash -l 
#PBS -l nodes=1:ppn=24
#PBS -l walltime=00:05:00
#PBS -m abe
#PBS -q gpu
#PBS -d .

#LAUNCHER='mpirun --bind-to none -np 1' regent.py ~/wic/wavelet.rg -i \
#~/wic/images/gates.png -ll:cpu 4 -ll:csize 1024

LAUNCHER='mpirun --bind-to none -np 1' regent.py ~/wic/wavelet.rg -i \
~/wic/images/gates.png -ll:cpu 2 -ll:csize 1024 \
-hl:spy -logfile ./logs/wavelet_%.log

# This will be with profiling + spy
#LAUNCHER='mpirun --bind-to none -np 2' regent.py ~/wic/wavelet.rg -i \
#~/wic/images/gates.png -ll:cpu 4 -ll:csize 1024 -hl:prof 1 \
#-hl:spy -logfile wavelet_%.log
 

