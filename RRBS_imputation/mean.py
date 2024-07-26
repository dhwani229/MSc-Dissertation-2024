# THIS IS THE CODE FOR IMPUTATION USING MEAN IMPUTATION. THE CODE SHOWN IS ON TRAINING DATA ON HEALTHY V BENIGN BUT THE SAME CODE WAS 
# USED ON HEALTHY V BENIGN TEST AND VALIDATION DATA AS WELL AS TRAINING, TEST, AND VALIDATION SET ON HEALTHY VS MALIGNANT.

import os
import pandas as pd
from sklearn.impute import SimpleImputer

# change directories to where training, validation, and test files are stored
os.chdir('/data/home/bt22912/files_needed/imputation/hb') 

# TRAINING DATA

train_df = pd.read_csv('train_hb.csv')

# Separate the columns from the 'Unnamed: 0' column (this is the first column that contains all the DMPs)
train_features = train_df.drop(columns=['Unnamed: 0'])

# Imputation using mean and SimpleImputer on scikit learn
imputer = SimpleImputer(strategy='mean')
train_features_imputed = imputer.fit_transform(train_features)
train_df_imputed = pd.DataFrame(train_features_imputed, columns=train_features.columns)

# Add the first column back
train_df_imputed.insert(0, 'Unnamed: 0', train_df['Unnamed: 0'])

train_df_imputed.to_csv('train_hb_mean.csv', index=False)
print(train_df_imputed.head())

# Check if the imputed dataset contained any negative values or values over 100
negative_values = (train_df_imputed.drop(columns=['Unnamed: 0']) < 0).any().any()
values_over_100 = (train_df_imputed.drop(columns=['Unnamed: 0']) > 100).any().any()

if negative_values:
    print("The imputed data contains negative values.")
else:
    print("The imputed data does not contain negative values.")

if values_over_100:
    print("The imputed data contains values over 100.")
else:
    print("The imputed data does not contain values over 100.")
