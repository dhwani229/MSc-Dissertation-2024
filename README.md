# MSc-Dissertation-2024

### Overview 

This project aims to investigate this effect using publicly available epigenomic, genomic, and transcriptomic datasets from cohorts of participants belonging into one of two clearly defined disease states; healthy or cancer. Testing how the different imputation methods will affect the supervised machine learning models’ efficiency will give us insight into the most appropriate bioinformatic approach for future studies aiming at the detection of disease status or the prediction of disease risk based on molecular biomarkers.

### Methodology
* Search in public databases for epigenomic, genomic, and transcriptomic data from cohorts of two distinct disease states (healthy vs cancer)
* Preprocessing of raw data in order to attain all relevant features (methylation levels, SNPs, CNVs, gene expression levels) per individual
* Missing value imputation based on multiple methodologies (kNN, random forest, neural networks etc) Feature selection and machine learning predictor algorithms development
* Assessment of the optimal imputation method based on the performance of the machine learning predictor algorithms.

I have found 3 datasets: 
* WGBS
* RRBS
* EPIC microarray

## PRE-PROCESSING
### RRBS job scripts explained

* js.sh
  * Job script for downloading the RRBS data from NCBI SRA using the software SRATOOLS
  * To download all samples in the dataset (beginning with SRR), I downloaded the txt file containing all sample names and made a code that looped through this txt file to obtain all samples from the dataset

* fq.sh
  * Job script to convert the downloaded .sra files into .fastq
  * Run time max was set to 48h and as a result, not all samples were converted, therefore I had to run another script to convert the rest of the samples. The rest of them were done in fq2.sh
  * Only worked up to sample SRR6350430

* fq2.sh
  * Job script to convert the rest of the .sra files into .fastq
  * The samples that weren't converted were from SRR6350431 to SRR6435636

* rm_sra.sh
  * Job script to remove the .sra files from all the directories to save space once the .fastq conversion was complete

* fastqc.sh
  * Quality check on each sample using software 'FASTQC'
  * The output files were all checked manually

* paired_fqscreen.sh
  * Once the samples were checked, I performed FastQ screen to determine the directionality of the samples
  * Samples were split into paired and unpaired
  * Paired samples: SRR6350255-SRR6350383, SRR6435611, SRR6435613, SRR6435615, SRR6435618
  * FastQ screen was downloaded via git clone and configured using hg38 Bisulfite converted genome (hg38 downloaded from: https://www.ncbi.nlm.nih.gov/genome/guide/human/ with chromosomal annotations and Bisulfite conversion done by Bismark)

* unpaired_fqscreen.sh
  * Similar to above but this job script focussed on the unpaired samples
  * Unpaired samples: SRR6350384-SRR6350433, SRR6435612, SRR6435614, SRR6435616, SRR6435617, SRR6435619-SRR6435636

* rm_fqc.sh
  * Removed fastqc output files once the quality checks were done

* rm_fqscreen.sh
   * Removal of fastq screen output files once analysed
   * All samples were directional

