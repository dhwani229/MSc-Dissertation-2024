# Fastqscreen on paired samples to work out directionality of samples before proceeding to trimming. Paired samples: SRR6350255-SRR6350383, SRR6435611, SRR6435613, SRR6435615, SRR6435618

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=240:00:00
#$ -l h_vmem=20G

module load bismark


FASTQ_SCREEN_PATH="/data/home/bt22912/temp/FastQ-Screen/fastq_screen"
CONFIG_FILE="/data/home/bt22912/temp/FastQ-Screen/fastq_screen.conf"
FASTQ_BASE_DIR="/data/scratch/bt22912/prac"


PAIRED_DIRS=( $(seq -f "SRR6350%g" 255 383) SRR6435611 SRR6435613 SRR6435615 SRR6435618 )

run_fastq_screen() {
    local dir=$1
    echo "Processing directory: $dir"


    cd "${FASTQ_BASE_DIR}/${dir}"


    for file in *_1.fastq; do
        if [ -e "$file" ]; then
            echo "Processing file: $file"
            $FASTQ_SCREEN_PATH --conf $CONFIG_FILE --bisulfite "$file"
        else
            echo "File not found: $file"
        fi
    done
}


for dir in "${PAIRED_DIRS[@]}"; do
    run_fastq_screen "$dir"
done
