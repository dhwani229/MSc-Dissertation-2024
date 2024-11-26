# THIS IS THE CODE TO CONVERT THE .SRA FILES INTO .FASTQ. 
# FOR THE NEXT SAMPLES. FASTQ CONVERSION TAKES TOO LONG THEREFORE TRY WITH ARRAYS

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=240:00:00
#$ -l h_vmem=15G
#$ -m bea
#$ -t 3-10

module load sratools/2.10.8

filepath=/data/scratch/bt22912/wgbs/f10.txt 
path="/data/scratch/bt22912/wgbs"
REPCORES=$((NSLOTS / 2))

number=${SGE_TASK_ID}
SRA_DIR=$(sed -n "${number}p" "$filepath")
    cd ${path}/${SRA_DIR}/
    
    fasterq-dump "${SRA_DIR}" --split-files --threads ${NSLOTS}
    
    
    cd ${path}
