# RMSE for kNN imputation. Only calculated on training data. missing_rate corresponds to % of missingness in filtered dataframe. 
# This e.g. is for HB, same code was used for HM but missing rate in HM = 20% and in HB = 30%

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
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

# nearest neighbours range for kNN to see which produces the best RMSE
neighbour_numbers = [1, 3, 5, 7, 9, 11,13,15,17,19,21]
rmse_scores = []

# loop through each neighbours value
for n_neighbours in neighbour_numbers:
    # Initialise loop
    imputer = KNNImputer(n_neighbours=n_neighbors)
    
    # fit and transform the data with introduced missing values
    imputed_data = imputer.fit_transform(missing_data)
    
    # create a new data frame with the imputed data
    imputed_df = pd.DataFrame(imputed_data, columns=data.columns)
    
    # concatenate the metadata with the imputed data
    imputed_df = pd.concat([metadata.reset_index(drop=True), imputed_df.reset_index(drop=True)], axis=1)
    
    # extract the values from the original data frame and the imputed data frame at the missing positions
    true_values = data.values[mask]
    imputed_values = imputed_data[mask]
    
    # Ensure no presence of NaNs in the true_values and imputed_values
    valid_mask = ~np.isnan(true_values) & ~np.isnan(imputed_values)
    true_values = true_values[valid_mask]
    imputed_values = imputed_values[valid_mask]
    
    # calculate RMSE, ensuring we only compare the originally missing values
    rmse = mean_squared_error(true_values, imputed_values, squared=False)
    rmse_scores.append((n_neighbours, rmse))
    print(f'RMSE for {n_neighbours} neighbors: {rmse}')

# number of neighbours with the lowest RMSE
optimal_n_neighbours, lowest_rmse = min(rmse_scores, key=lambda x: x[1])
print(f'Optimal number of neighbours: {optimal_n_neighbours} with RMSE: {lowest_rmse}')
