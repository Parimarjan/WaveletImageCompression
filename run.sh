#!/bin/bash -l 
#PBS -l nodes=1:ppn=24
#PBS -l walltime=00:05:00
#PBS -m abe
#PBS -q gpu
#PBS -d .

LAUNCHER='mpirun --bind-to none -np 2' regent.py ~/project/wavelet.rg -i \
~/project/images/gates.png -o ~/project/test.png -ll:cpu 4 -ll:csize 1024 -hl:prof 1 \
-hl:spy -logfile wavelet_parallel

 

