# methylation calling on all samples 

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 6
#$ -l h_rt=240:00:00
#$ -l h_vmem=4G  
#$ -m a
#$ -t 1-205

module load bismark
module load samtools

filepath=/data/scratch/bt22912/RRBS.txt 
path=/data/scratch/bt22912/prac
PAIRED_DIRS=( $(seq -f "SRR6350%g" 255 383) SRR6435611 SRR6435613 SRR6435615 SRR6435618 )
REPCORES=$((NSLOTS / 3))

number=${SGE_TASK_ID}
SRA_DIR=$(sed -n "${number}p" "$filepath")
	cd ${path}/${SRA_DIR}/alignment/

		if [[ " ${PAIRED_DIRS[@]} " =~ " ${SRA_DIR} " ]]; then
			bismark_methylation_extractor --gzip --bedGraph --buffer_size 10G --multicore 2 *_1_val_1_bismark_bt2_pe.bam -o ${path}/${SRA_DIR}/methylation/


		else 

			bismark_methylation_extractor --gzip --bedGraph --buffer_size 10G --multicore 2 *_trimmed_bismark_bt2.bam -o ${path}/${SRA_DIR}/methylation/

		fi 

	cd ${path} 
