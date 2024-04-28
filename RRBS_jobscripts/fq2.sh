# jobscript for samples SRR6350431 to SRR6435636 (previous jobscript ended before it could finish these) 

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=240:00:00
#$ -l h_vmem=8G

module load sratools/2.10.8

path="/data/scratch/bt22912/prac"


start=6350431
end=6435636

for i in $(seq $start $end); do
    SRA_DIR="SRR$i"
    
    
    if [ -d "${path}/${SRA_DIR}" ]; then
        cd "${path}/${SRA_DIR}"
        
        
        fasterq-dump "${SRA_DIR}" --split-files --threads 1
        
        
        cd ..
    else
        echo "Directory ${path}/${SRA_DIR} does not exist."
    fi
done
