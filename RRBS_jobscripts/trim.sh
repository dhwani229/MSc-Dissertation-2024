# trimming the fastq files and deleting the fastq files after trimming.

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:00:00
#$ -l h_vmem=25G
#$ -m bea

module load trimgalore
path=/data/scratch/bt22912/prac
PAIRED_DIRS=( $(seq -f "SRR6350%g" 255 383) SRR6435611 SRR6435613 SRR6435615 SRR6435618 )

for SRA_DIR in $(ls ${path}); do
    
    if [[ " ${PAIRED_DIRS[@]} " =~ " ${SRA_DIR} " ]]; then
        cd ${path}/${SRA_DIR}/
        trim_galore --cores ${NSLOTS} --paired --rrbs --fastqc *_1.fastq *_2.fastq -o trimming/
        if [ $? -eq 0 ]; then
            echo "Trimming successful"
            rm -v *_1.fastq *_2.fastq
        else
            echo "Trimming failed"
        fi
        cd ..
    else
        cd ${path}/${SRA_DIR}/
        trim_galore --cores ${NSLOTS} --rrbs --fastqc *.fastq -o trimming/ 
        if [ $? -eq 0 ]; then
            echo "Trimming successful"
            rm -v *.fastq
        else
            echo "Trimming failed"
        fi
        cd ..
    fi
done
