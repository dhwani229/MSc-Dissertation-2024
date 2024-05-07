# alignment of sample fq files to reference genome 

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:00:00
#$ -l h_vmem=5G  
#$ -m bea

module load bismark


path="/data/scratch/bt22912/prac"
REF_GENOME_PATH="/data/scratch/bt22912/dissertation/ref_genome/GRCh38_chr"
PAIRED_DIRS=( $(seq -f "SRR6350%g" 255 383) SRR6435611 SRR6435613 SRR6435615 SRR6435618 )
REPCORES=$((NSLOTS / 2))

for SRA_DIR in $(ls ${path} | grep SRR); do
    cd ${path}/${SRA_DIR}/trimming/
    
    
    if [[ " ${PAIRED_DIRS[@]} " =~ " ${SRA_DIR} " ]]; then
        
        bismark --genome ${REF_GENOME_PATH} -p ${REPCORES} -1 *_1_val_1.fq -2 *_2_val_2.fq -o ${path}/${SRA_DIR}/alignment/
        
        if [ $? -eq 0 ]; then
            echo "Alignment successful"
            rm -v *_1_val_1.fq *_2_val_2.fq
        else
            echo "Alignment failed" 
        fi
    else
        
        bismark --genome ${REF_GENOME_PATH} -p ${REPCORES} *_trimmed.fq -o ${path}/${SRA_DIR}/alignment/ 
       
        if [ $? -eq 0 ]; then
            echo "Alignment successful"
            rm -v *_trimmed.fq
        else
            echo "Alignment failed" 
        fi
    fi
    
    
    cd ${path}
done
