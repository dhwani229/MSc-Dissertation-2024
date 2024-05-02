# fastqscreen to work out directionality of unpaired samples before proceeding to trimming. unpaired samples: SRR6350384-SRR6350433, SRR6435612, SRR6435614, SRR6435616, SRR6435617, SRR6435619-SRR6435636

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=240:00:00
#$ -l h_vmem=25G

module load bismark

FASTQ_SCREEN_PATH="/data/home/bt22912/temp/FastQ-Screen/fastq_screen"
CONFIG_FILE="/data/home/bt22912/temp/FastQ-Screen/fastq_screen.conf"
FASTQ_BASE_DIR="/data/scratch/bt22912/prac"


UNPAIRED_DIRS=( $(seq -f "SRR6350%g" 384 433) $(seq -f "SRR64356%g" 19 36) SRR6435612 SRR6435614 SRR6435616 SRR6435617 )


run_fastq_screen() {
    local dir=$1
    echo "Processing directory: $dir"

    
    cd "${FASTQ_BASE_DIR}/${dir}" || exit

    
    echo "Listing files in $(pwd):"
    ls -l

    
    for file in *.fastq; do
        if [ -e "$file" ]; then
            echo "Processing file: $file"
            $FASTQ_SCREEN_PATH --conf $CONFIG_FILE --bisulfite "$file"
        else
            echo "File not found: $file"
        fi
    done
}


for dir in "${UNPAIRED_DIRS[@]}"; do
    run_fastq_screen "$dir"
done
