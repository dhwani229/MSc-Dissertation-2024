# jobscript to convert the .sra files into .fastq 

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:00:00
#$ -l h_vmem=15G
#$ -m bea

module load sratools/2.10.8

#for SRA_DIR in SRR*/
path=/data/scratch/bt22912/prac

for SRA_DIR in $(ls ${path} | grep SRR)
do
    cd ${path}/${SRA_DIR}/   
    
    fasterq-dump "${SRA_DIR}" --split-files --threads ${NSLOTS}    

    
    cd ..
done
