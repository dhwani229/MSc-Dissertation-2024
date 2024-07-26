#TRAIN TEST SPLIT FOR HEALTHY AND BENIGN. SAME CODE WAS USED FOR HEALTHY V MALIGNANT

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# change directory to where files are locatedd
os.chdir('/data/home/bt22912/files_needed') 

# define samples 
healthy_samples = ['SRR6350325',
'SRR6350326',
'SRR6350327',
'SRR6350328',
'SRR6350329',
'SRR6350330',
'SRR6350331',
'SRR6350332',
'SRR6350333',
'SRR6350334',
'SRR6350335',
'SRR6350336',
'SRR6350337',
'SRR6350338',
'SRR6350339',
'SRR6350340',
'SRR6350341',
'SRR6350342',
'SRR6350343',
'SRR6350344',
'GSM2877445',
'SRR6350347',
'SRR6350348',
'SRR6350349',
'SRR6350350',
'SRR6350351',
'SRR6350352',
'SRR6350353',
'SRR6350354',
'SRR6350355',
'SRR6350356',
'SRR6350357',
'SRR6350358',
'SRR6350359',
'SRR6350360',
'SRR6350361',
'SRR6350362',
'SRR6350363',
'SRR6350364',
'SRR6350365',
'SRR6350366',
'SRR6350367']

benign_samples = ['SRR6435624',
'SRR6435625',
'GSM2910007',
'SRR6435629',
'SRR6435630',
'SRR6435631',
'SRR6435632',
'SRR6435633',
'SRR6435634',
'SRR6435635',
'SRR6435636',
'SRR6350297',
'SRR6350298',
'SRR6350299',
'SRR6350300',
'SRR6350301',
'SRR6350302',
'SRR6350303',
'SRR6350304',
'SRR6350305',
'SRR6350306',
'SRR6350307',
'SRR6350308',
'SRR6350309',
'SRR6350310',
'SRR6350311',
'SRR6350312',
'SRR6350313',
'SRR6350314',
'SRR6350315',
'SRR6350316',
'SRR6350317',
'SRR6350318',
'SRR6350319',
'SRR6350320',
'SRR6350321',
'SRR6350322',
'SRR6350323',
'SRR6350324',
'SRR6350402',
'SRR6350403',
'SRR6350404',
'GSM2877501',
'SRR6350407',
'GSM2877503',
'SRR6350410',
'SRR6350411',
'SRR6350412',
'GSM2877507',
'GSM2877508',
'SRR6350418',
'SRR6350419',
'SRR6350420',
'SRR6350421',
'SRR6350422',
'SRR6350423',
'SRR6350424',
'SRR6350425',
'GSM2877517',
'SRR6350428',
'SRR6350429',
'SRR6350430',
'SRR6350431',
'SRR6350432']

# SPLITTING FILES INTO TRAINING, TEST, VALIDATION

df = pd.read_csv('filtered_df_hb.csv')
chromosomal_positions = df.iloc[:, 0]

# making classes to allow for a stratified split of healthy and benign samples in each set
sample_class = {sample: 'healthy' for sample in healthy_samples}
sample_class.update({sample: 'benign' for sample in benign_samples})

# create dataframe for the classes
class_df = pd.DataFrame(list(sample_class.items()), columns=['sample_id', 'class'])

# transpose the DataFrame, excluding the DMPs to bring sample id as rows (were originally columns) for train_test_split function
df_transposed = df.iloc[:, 1:].transpose().reset_index()
df_transposed.columns = ['sample_id'] + list(chromosomal_positions)

# merge with class dataframe
df_with_class = df_transposed.merge(class_df, on='sample_id')

# split into training (60%) and temporary (40%) sets
train, temp = train_test_split(df_with_class, test_size=0.4, random_state=42, stratify=df_with_class['class'])

# split the temporary set into validation (50% of temporary, 20% of total) and test (50% of temporary, 20% of total) sets = gives a 60/20/20 split
validation, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['class'])

train_df = train.drop(columns=['class']).set_index('sample_id').transpose()
val_df = validation.drop(columns=['class']).set_index('sample_id').transpose()
test_df = test.drop(columns=['class']).set_index('sample_id').transpose()

# add back the DMPs as the first column
train_df.insert(0, 'Unnamed: 0', chromosomal_positions.values)
val_df.insert(0, 'Unnamed: 0', chromosomal_positions.values)
test_df.insert(0, 'Unnamed: 0', chromosomal_positions.values)

train_df.set_index('Unnamed: 0', inplace=True)
val_df.set_index('Unnamed: 0', inplace=True)
test_df.set_index('Unnamed: 0', inplace=True)

print(train_df)
print(val_df)
print(test_df)

# save the splits to CSV files
#train_df.to_csv('train_hb.csv')
#val_df.to_csv('val_hb.csv')
#test_df.to_csv('test_hb.csv')
