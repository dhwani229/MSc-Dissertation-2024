# ALIGN TRIMMED SAMPLES TO BISULFITE CONVERTED GENOME 

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 10
#$ -l h_rt=240:00:00
#$ -l h_vmem=10G  
#$ -m bea
#$ -t 3-10

module load bismark
module load samtools

filepath=/data/scratch/bt22912/wgbs/f10.txt 
path="/data/scratch/bt22912/wgbs"
REF_GENOME_PATH=/data/home/bt22912/genome/ref

REPCORES=$((NSLOTS / 2))

number=${SGE_TASK_ID}
SRA_DIR=$(sed -n "${number}p" "$filepath")
    cd ${path}/${SRA_DIR}/trimming/
        
        bismark --genome ${REF_GENOME_PATH} -p 4 -1 *_1_val_1.fq -2 *_2_val_2.fq -o ${path}/${SRA_DIR}/alignment/
        
    
    
    cd ${path}
