# filtering the original csv using the hg38_converted bed file to show only the chromosomal positions that were of interest in the 
# original article.

import pandas as pd

processed_df = pd.read_csv('processed_healthyvbenign.csv') # replace with processed_healthyvmalignant.csv for HM 
bed_df = pd.read_csv('hg38_converted.bed', sep='\t', header=None, names=['chrom', 'start', 'end'])

# extract the chromosome and position from processed_df index
processed_df[['chrom', 'start', 'end']] = processed_df.iloc[:, 0].str.split('.', expand=True)
processed_df['start'] = processed_df['start'].astype(int)
processed_df['end'] = processed_df['end'].astype(int)

# initialise a list to collect filtered rows
filtered_rows = []

# iterate through each row in the bed_df and filter the processed_df 
for index, row in bed_df.iterrows():
    chrom = row['chrom']
    start = row['start']
    end = row['end']
    
    # filter the processed_df
    filtered = processed_df[(processed_df['chrom'] == chrom) &
                            (processed_df['start'] >= start) &
                            (processed_df['end'] <= end)]
    
    # append the filtered rows to the list
    filtered_rows.append(filtered)

# concatenate all filtered rows into a single dataframe
filtered_df = pd.concat(filtered_rows).drop(columns=['chrom', 'start', 'end'])

# Save the filtered dataframe to a new CSV file
# filtered_df.to_csv('filtered_df_hb.csv', index=False)

print(filtered_df.head())
