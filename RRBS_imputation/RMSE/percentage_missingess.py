# percentage of missing values in HB and HM csv. This was used as the value for missing_rate in the RMSE calculations.
# HB % missing = 25.61% therefore missing_rate=0.3 in RMSE calculation
# HM % missing = 17.99% therefore missing_rate=0.2 in RMSE calculation

# percentage of NaN values 
import os
import pandas as pd

os.chdir('/data/home/bt22912/files_needed/imputation/hb')
nan_perc=pd.read_csv('train_hb.csv') # replace with train_hm.csv for HM

# Calculate the total number of elements in the DataFrame
total_elements = nan_perc.size

# Calculate the total number of NaN values in the DataFrame
total_nans = nan_perc.isna().sum().sum()

# Calculate the percentage of NaN values
percentage_nans = (total_nans / total_elements) * 100

print(f"Percentage of NaN values: {percentage_nans:.2f}%")
