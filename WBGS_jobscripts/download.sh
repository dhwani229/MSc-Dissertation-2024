#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=240:00:00
#$ -l h_vmem=40G
#$ -m bea

module load sratools/2.10.8

while IFS= read -r srr_id
do
  echo "Downloading: $srr_id"
  prefetch "$srr_id"  
done < /data/scratch/bt22912/wgbs/f10.txt #f10.sh = file containing the names of the first 10 samples. Full sample list contained in the file called 'wgbs.txt'
[bt22912@frontend11 wgbs]$ 
