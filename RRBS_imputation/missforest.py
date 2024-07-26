# THIS IS THE CODE FOR IMPUTATION USING MISSFOREST IMPUTATION. THE CODE SHOWN IS ON TRAINING DATA ON HEALTHY V BENIGN BUT THE SAME CODE WAS 
# USED ON HEALTHY V BENIGN TEST AND VALIDATION DATA AS WELL AS TRAINING, TEST, AND VALIDATION SET ON HEALTHY VS MALIGNANT.

import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# change directories to where training, validation, and test files are stored
os.chdir('/data/home/bt22912/files_needed/imputation/hb') 

# TRAINING DATA

train_df = pd.read_csv('train_hb.csv')

# Separate the columns from the 'Unnamed: 0' column (this is the first column that contains all the DMPs)
train_features = train_df.drop(columns=['Unnamed: 0'])

# TRAINING DATA

# Define the Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Imputation using IterativeImputer and random forest regressor
imputer = IterativeImputer(estimator=rf_regressor, max_iter=2, random_state=42)
train_features_imputed = imputer.fit_transform(train_features)

# Add the first column back
train_df_missforest = pd.DataFrame(train_features_imputed, columns=train_features.columns)
train_df_missforest.insert(0, 'Unnamed: 0', train_df['Unnamed: 0'])

train_df_missforest.to_csv('train_hb_missforest.csv', index=False)


print("Imputed Training Data Sample:")
print(train_df_missforest.head())


# Check if the imputed dataset contained any negative values or values over 100
negative_values = (train_df_missforest.drop(columns=['Unnamed: 0']) < 0).any().any()
values_over_100 = (train_df_missforest.drop(columns=['Unnamed: 0']) > 100).any().any()

if negative_values:
    print("The imputed data contains negative values.")
else:
    print("The imputed data does not contain negative values.")

if values_over_100:
    print("The imputed data contains values over 100.")
else:
    print("The imputed data does not contain values over 100.")
