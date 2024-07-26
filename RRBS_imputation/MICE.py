# THIS IS THE CODE FOR IMPUTATION USING MICE IMPUTATION. THE CODE SHOWN IS ON TRAINING DATA ON HEALTHY V BENIGN BUT IT PRODUCED 
# NEGATIVE RESULTS AND RESULTS OVER 100 THEREFORE DIDNT PROCEED ANY FURTHER 

import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# change directories to where training, validation, and test files are stored
os.chdir('/data/home/bt22912/files_needed/imputation/hb') 

# TRAINING DATA

train_df = pd.read_csv('train_hb.csv')

# Separate the columns from the 'Unnamed: 0' column (this is the first column that contains all the DMPs)
train_features = train_df.drop(columns=['Unnamed: 0'])

# initialising iterativeimputer from scikit learn
imputer = IterativeImputer(max_iter=10, random_state=0)

# imputation using MICE 
train_imputed = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
print(train_imputed)

# Check if the imputed dataset contained any negative values or values over 100
negative_values = (train_imputed < 0).any().any()
values_over_100 = (train_imputed > 100).any().any()

if negative_values:
    print("The imputed data contains negative values.") 
else:
    print("The imputed data does not contain negative values.")

if values_over_100:
    print("The imputed data contains values over 100.")
else:
    print("The imputed data does not contain values over 100.")
