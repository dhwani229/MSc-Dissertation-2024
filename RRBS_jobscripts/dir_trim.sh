# make a directory called trimming inside each sample directory

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=1:00:00
#$ -l h_vmem=1G

path=/data/scratch/bt22912/prac


for SRA_DIR in $(ls ${path} | grep SRR)
do
    
    cd ${path}/${SRA_DIR}/
    
    mkdir trimming
    
    
    cd ..
done
