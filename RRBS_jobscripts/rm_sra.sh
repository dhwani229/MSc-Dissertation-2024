# removing .sra files from all directories to save space

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=240:00:00
#$ -l h_vmem=10G

path=/data/scratch/bt22912/prac


for SRA_DIR in $(ls ${path} | grep SRR)
do

    echo cd ${path}/${SRA_DIR}/


    echo rm -v *.sra


    cd ..
done
