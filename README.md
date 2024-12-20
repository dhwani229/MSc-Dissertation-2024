# MSc-Dissertation-2024

### Overview 

This project aims to investigate this effect using publicly available epigenomic, genomic, and transcriptomic datasets from cohorts of participants belonging into one of two clearly defined disease states; healthy or cancer. Testing how the different imputation methods will affect the supervised machine learning models’ efficiency will give us insight into the most appropriate bioinformatic approach for future studies aiming at the detection of disease status or the prediction of disease risk based on molecular biomarkers.

### Methodology
* Search in public databases for epigenomic, genomic, and transcriptomic data from cohorts of two distinct disease states (healthy vs cancer)
* Preprocessing of raw data in order to attain all relevant features per individual
* Missing value imputation based on multiple methodologies (kNN, random forest etc) Feature selection and machine learning predictor algorithms development
* Assessment of the optimal imputation method based on the performance of the machine learning predictor algorithms.

2 datasets were acquired: 
* WGBS
* RRBS


# PRE-PROCESSING
## RRBS job scripts explained

* js.sh
  * Job script for downloading the RRBS data from NCBI SRA using the software SRATOOLS
  * To download all samples in the dataset (beginning with SRR), a txt file containing all sample names was downloaded and a code that looped through this txt file to obtain all samples from the dataset was made.

* fq.sh
  * Job script to convert the downloaded .sra files into .fastq

* rm_sra.sh
  * Job script to remove the .sra files from all the directories to save space once the .fastq conversion was complete

* fastqc.sh
  * Quality check on each sample using software 'FASTQC'
  * The output files were all checked manually

* paired_fqscreen.sh
  * Once the samples were checked, FastQ screen was used to determine the directionality of the samples
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
 
 * trim.sh
   * Trimming of fastq files using 'Trim Galore!'
   * Used additional option '--fastqc' to produce quality reports
   * Also removed the original fastq IF trimming was successful and preserved the files if trimming failed. This was done to save space
   * Manually checked the fastqc output files to make sure the trimmed files were appropriate for use in the next stage
  
 * align.sh
   * Alignment, using Bismark, to a bisulfite-converted hg38 reference genome (bisulfite conversion done by bismark)

* efficiency.sh
  * obtaining methylation efficiency from txt report from the alignment and putting all of them into one txt file.
  * anything below 50% efficiency = bad and was discarded

* methylation.sh
  * methylation calling using bismark methylation extractor
  * provided coverage files specific to CpG coverage
 

### Processing of duplicate RRBS samples 

Some of the raw RRBS SRA data corresponded to the same samples (contained under the sample IDs beginning with 'GSM') and to process these, they were combined them into one fq file since there was no specific explanation on the article about the reason for there being duplicate samples present. Therefore, I assumed that these were just bad sequencing runs and decided to combine them.

* download.sh
  * Making the directories, called GSMxxx, where the corresponding SRA runs will be stored 
  * Creating a text file that contains the GSM ID and its corresponding SRA runs
  * Using 'prefetch' from sratools to download the required SRA files

* fq.sh
  * Converting the .sra files to .fq

using the 'cat' command on the required samples, they were concatenated and then from this point forward, the same job scripts as the regular RRBS were used. 

## WGBS job scripts explained

WGBS samples were processed in smaller batches of 10 samples due to storage limitations. The processing was similar to the RRBS samples but instead of directionality determination, WGBS samples needed to be deduplicated before methylation extraction. 
