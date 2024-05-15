# convert all .sra files to .fq 

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:00:00
#$ -l h_vmem=15G
#$ -m bea

module load sratools/2.10.8

path=/data/scratch/bt22912/duplicate_rrbs

for sample_dir in $(ls $path | grep GSM)
do
    cd ${path}/${sample_dir}/ 

    
    for SRA_DIR in $(ls | grep SRR)
    do
        cd ${SRA_DIR}/ 

        fasterq-dump ${SRA_DIR} --split-files --threads ${NSLOTS}
        

        cd ..  
    done
    cd ..  
done
