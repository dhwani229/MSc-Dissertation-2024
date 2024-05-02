# removing fastqc files after checking

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=240:00:00
#$ -l h_vmem=8G

path=/data/scratch/bt22912/prac


for SRA_DIR in $(ls ${path} | grep SRR)
do
    
    cd ${path}/${SRA_DIR}/
    
    
    rm -v *_fastqc.html *_fastqc.zip
    
    
    cd ..
done
