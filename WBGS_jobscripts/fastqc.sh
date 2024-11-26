# FASTQC ON THE FASTQ FILES TO DETERMINE QUALITY OF SAMPLES

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 10
#$ -l h_rt=48:00:00
#$ -l h_vmem=3G
#$ -m bea

module load fastqc
path=/data/scratch/bt22912/wgbs

for SRA_DIR in $(ls ${path} | grep SRR)
do
    cd ${path}/${SRA_DIR}/

    fastqc *.fastq -o fqc/
    cd ..
done

