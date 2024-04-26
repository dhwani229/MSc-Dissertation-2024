# jobscript for downloading RRBS data from SRA using sratools

#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=48:00:00
#$ -l h_vmem=4G

module load sratools/2.10.8

while IFS= read -r srr_id
do
  echo "Downloading: $srr_id"
  prefetch "$srr_id"  
done < /data/scratch/bt22912/practice/RRBS.txt

echo "All downloads completed!"
