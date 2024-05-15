# obtaining methylation efficiencies from alignment output txt report and compiling them into one text file 

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=1:00:00
#$ -l h_vmem=1G  
#$ -m bea

path="/data/scratch/bt22912/prac"
output_file="${path}/mapping_efficiencies.txt"


echo "Sample Name,Mapping Efficiency" > $output_file

for SRA_DIR in $(ls ${path} | grep SRR)
do
    
    cd ${path}/${SRA_DIR}/alignment/

alignment_report=$(find $alignment_folder -type f -name "*.txt" -print -quit)
    
    if [ -f "$alignment_report" ]; then
        # Extract the Mapping Efficiency line
        efficiency_line=$(grep -i "Mapping efficiency" "$alignment_report")
        if [[ ! -z $efficiency_line ]]; then
            efficiency_score=$(echo $efficiency_line | grep -o -E '[0-9]+(\.[0-9]+)?%')
            echo "${SRA_DIR},${efficiency_score}" >> $output_file
        else
            echo "${SRA_DIR},No efficiency score found" >> $output_file
        fi
    else
        echo "${SRA_DIR},No report file found" >> $output_file
    fi
done
