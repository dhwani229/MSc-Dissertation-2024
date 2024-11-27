# METHYLATION EXTRACTION

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=240:00:00
#$ -l h_vmem=12G  
#$ -m bea
#$ -t 1-2

module load bismark
module load samtools

filepath=/data/scratch/bt22912/wgbs/f10.txt 
path=/data/scratch/bt22912/wgbs
REPCORES=$((NSLOTS / 3))

number=${SGE_TASK_ID}
SRA_DIR=$(sed -n "${number}p" "$filepath")
	cd ${path}/${SRA_DIR}/deduplication/

		bismark_methylation_extractor --gzip --bedGraph --buffer_size 16G --multicore 4 *.deduplicated.bam -o ${path}/${SRA_DIR}/methylation/

	cd ${path}
