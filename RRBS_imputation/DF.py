# python code 

# ORIGINAL DF

import os
import pandas as pd


os.chdir('/data/scratch/bt22912/prac') 


print("New Working Directory:", os.getcwd())


original_file_path = 'processed_healthyvbenign.csv'
df = pd.read_csv(original_file_path, index_col=0)
print(df.head())

# BED_DF WITH COLUMN TITLES  (original bed file didnt have any)

bed_file = 'hg38_converted.bed'
bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end'])


print(bed_df.head())

# FILTERED DF

processed_df = pd.read_csv('processed_healthyvbenign.csv')
bed_df = pd.read_csv('hg38_converted.bed', sep='\t', header=None, names=['chrom', 'start', 'end'])

# Extract the chromosome and position information from the processed_df index
processed_df[['chrom', 'start', 'end']] = processed_df.iloc[:, 0].str.split('.', expand=True)
processed_df['start'] = processed_df['start'].astype(int)
processed_df['end'] = processed_df['end'].astype(int)

# Initialize a list to collect filtered rows
filtered_rows = []

# Iterate through each row in the bed_df and filter the processed_df accordingly
for index, row in bed_df.iterrows():
    chrom = row['chrom']
    start = row['start']
    end = row['end']
    
    # Filter the processed_df
    filtered = processed_df[(processed_df['chrom'] == chrom) &
                            (processed_df['start'] >= start) &
                            (processed_df['end'] <= end)]
    
    # Append the filtered rows to the list
    filtered_rows.append(filtered)

# Concatenate all filtered rows into a single dataframe
filtered_df = pd.concat(filtered_rows).drop(columns=['chrom', 'start', 'end'])

# Save the filtered dataframe to a new CSV file
#filtered_df.to_csv('filtered_processed_healthyvbenign.csv', index=False)

# Show the filtered dataframe
print(filtered_df.head())
