# DEDUPLICATION OF SAMPLES AFTER ALIGNMENT. ONLY APPLICABLE TO WGBS SAMPLES.

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 10
#$ -l h_rt=240:00:00
#$ -l h_vmem=10G  
#$ -m bea
#$ -t 1-2

module load bismark
module load samtools

filepath=/data/scratch/bt22912/wgbs/f10.txt
path=/data/scratch/bt22912/wgbs

number=${SGE_TASK_ID}
SRA_DIR=$(sed -n "${number}p" "$filepath")
    	cd ${path}/${SRA_DIR}/alignment/

		deduplicate_bismark *.bam --output_dir ${path}/${SRA_DIR}/deduplication

	cd ..
