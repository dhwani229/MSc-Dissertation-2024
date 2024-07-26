# THIS IS THE CODE FOR IMPUTATION USING SVM IMPUTATION. THE CODE SHOWN IS ON TRAINING DATA ON HEALTHY V MALIGNANT BUT IT PRODUCED 
# NEGATIVE RESULTS THEREFORE DIDNT PROCEED ANY FURTHER 

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.svm import SVR

# Load the data
train_df = pd.read_csv('train_hm.csv')  

# removing first column (chromosomal positions) 
metadata = filtered_df.iloc[:, :1]  
data = filtered_df.iloc[:, 1:] 

# initialising iterativeimputer on scikit learn
imputer = IterativeImputer(estimator=SVR(), random_state=42, max_iter=10, n_nearest_features=None, initial_strategy='mean')

# imputation 
imputed_data = imputer.fit_transform(data)

imputed_df = pd.DataFrame(imputed_data, columns=data.columns)

# adding first column back
imputed_df = pd.concat([metadata.reset_index(drop=True), imputed_df.reset_index(drop=True)], axis=1)

# Check if the imputed dataset contained any negative values or values over 100
negative_values = (imputed_df.drop(columns=['Unnamed: 0']) < 0).any().any()
values_over_100 = (imputed_df.drop(columns=['Unnamed: 0']) > 100).any().any()

if negative_values:
    print("The imputed data contains negative values.")
else:
    print("The imputed data does not contain negative values.")

if values_over_100:
    print("The imputed data contains values over 100.")
else:
    print("The imputed data does not contain values over 100.")
