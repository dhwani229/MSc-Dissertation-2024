# RMSE for mode imputation. Only calculated on training data. missing_rate corresponds to % of missingness in filtered dataframe. 
# This e.g. is for HB, same code was used for HM but missing rate in HM = 20% and in HB = 30%

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def introduce_missing_values(df, missing_rate=0.3, seed=42):
    np.random.seed(seed)
    df_missing = df.copy()
    mask = np.random.rand(*df_missing.shape) < missing_rate
    df_missing[mask] = np.nan
    return df_missing, mask

train_df = pd.read_csv('train_hb.csv') # would be train_hm.csv for the HM data

# Separate the feature columns from the 'Unnamed: 0' column
train_features = train_df.drop(columns=['Unnamed: 0'])

# remove any NaNs in original data
original_na_mask = train_features.isna()

# artificial missingness introduction
missing_rate = 0.3
train_features_missing, mask = introduce_missing_values(train_features, missing_rate=missing_rate, seed=42)

# Combine the original NaNs mask with the new missing values mask
combined_mask = mask & ~original_na_mask

# imputation using mode ('most_frequent' on scikit learn)
imputer = SimpleImputer(strategy='most_frequent')
train_features_imputed = imputer.fit_transform(train_features_missing)

# Flatten arrays for RMSE calculation
original_values = train_features.values[combined_mask]
imputed_values = train_features_imputed[combined_mask]


# RMSE calculation between original and artifically imputed values
rmse = mean_squared_error(original_values, imputed_values, squared=False)
print(f'RMSE for mode imputation: {rmse}')
