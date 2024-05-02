# FastQC on samples for quality control

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=48:00:00
#$ -l h_vmem=10G

module load fastqc
path=/data/scratch/bt22912/prac

for SRA_DIR in $(ls ${path} | grep SRR)
do
    cd ${path}/${SRA_DIR}/

    fastqc *.fastq "${SRA_DIR}/results"
    cd ..
done
