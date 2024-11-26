# TRIMMING FILES WITH THE --fastqc FLAG TO GET FASTQC QUALITY REPORT OF THE TRIMMED SAMPLES AT THE SAME TIME

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:00:00
#$ -l h_vmem=25G
#$ -m bea
#$ -t 3-10

module load trimgalore

filepath=/data/scratch/bt22912/wgbs/f10.txt
path=/data/scratch/bt22912/wgbs
REPCORES=$((NSLOTS / 2))

number=${SGE_TASK_ID}
SRA_DIR=$(sed -n "${number}p" "$filepath")
	cd ${path}/${SRA_DIR}/
        	trim_galore --cores ${NSLOTS} --paired --fastqc *_1.fastq *_2.fastq -o trimming/
        	if [ $? -eq 0 ]; then
            		echo "Trimming successful, deleting original files."
            		echo rm -v *_1.fastq *_2.fastq
        	else
            		echo "Trimming failed, original files preserved."
        	fi
        cd ..C
