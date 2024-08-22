# RMSE for missforest imputation. Only calculated on training data. missing_rate corresponds to % of missingness in filtered dataframe. 
# This e.g. is for HB, same code was used for HM but missing rate in HM = 20% and in HB = 30%

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def introduce_missing_values(data, missing_rate=0.3, random_seed=42):
    np.random.seed(random_seed)
    mask = np.random.rand(*data.shape) < missing_rate
    data_missing = data.copy()
    data_missing[mask] = np.nan
    return data_missing, mask

train_df = pd.read_csv('train_hb.csv') # would be train_hm.csv for the HM data

# seperate the metadata (first column = chromosomal positions)
metadata = train_df.iloc[:, :1]  
data = train_df.drop(columns=['Unnamed: 0']) # Unnamed: 0 is the chromosomal positions

# articial missingness introduction
missing_data, mask = introduce_missing_values(data, missing_rate=0.3, random_seed=42)

# range of iteration numbers to test 
iteration_numbers = [1,2,3,4,5,6,7,8,9, 10]# 15, 20, 25, 30]
rmse_scores = []

# loop over each number
for n_iter in iteration_numbers:
    # Initialise using Iterative Imputer with RandomForestRegressor as the estimator
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    imputer = IterativeImputer(estimator=rf_regressor, random_state=42, max_iter=n_iter)
    
    # fit and transform the data with introduced missing values
    imputed_data = imputer.fit_transform(missing_data)
    
    # Create a new data frame with the imputed data
    imputed_df = pd.DataFrame(imputed_data, columns=data.columns)
    
    # concatenate the metadata back with the imputed data
    imputed_df = pd.concat([metadata.reset_index(drop=True), imputed_df.reset_index(drop=True)], axis=1)
    
    # extract the values from the original data frame and the imputed data frame at the missing positions
    true_values = data.values[mask]
    imputed_values = imputed_data[mask]
    
    # Ensure we don't have NaNs in the true_values and imputed_values
    valid_mask = ~np.isnan(true_values) & ~np.isnan(imputed_values)
    true_values = true_values[valid_mask]
    imputed_values = imputed_values[valid_mask]
    
    # calculate RMSE, ensuring we only compare the originally missing values
    rmse = mean_squared_error(true_values, imputed_values, squared=False)
    rmse_scores.append((n_iter, rmse))
    print(f'RMSE for {n_iter} iterations: {rmse}')

# Find the number of iterations with the lowest RMSE and print the values 
optimal_n_iter, lowest_rmse = min(rmse_scores, key=lambda x: x[1])
print(f'Optimal number of iterations: {optimal_n_iter} with RMSE: {lowest_rmse}')
