# original file from article containing DMPs of interest (hg19 genome). these were converted using UCSC liftover tool to produce hg38_converted.bed. this needed to be 
# converted to a file that was tab seperated and needed addition of headers 'chromsome', 'start', 'end' which would then be used for filtering

bed_file = 'hg38_converted.bed'
bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end'])

print(bed_df.head())
