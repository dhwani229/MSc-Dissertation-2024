# median imputation (manually)

# Calculate the median for each column, excluding NaNs
medians = filtered_df.median()

# Impute missing values with the median of each column
filtered_df_imputed_median = filtered_df.fillna(medians)

# Save the imputed dataframe to a new CSV file
#filtered_df_imputed_median.to_csv('filtered_processed_healthyvbenign_imputed_median.csv', index=False)

# Show the first few rows of the imputed dataframe
print(filtered_df_imputed_median.head())

# median imputation (SingleImputer)

import pandas as pd
from sklearn.impute import SimpleImputer

# Assuming filtered_df is already available

# Select only the numeric columns
numeric_data = filtered_df.select_dtypes(include=[float, int])

# Applying Median Imputation
median_imputer = SimpleImputer(strategy='median')
imputed_median_data = pd.DataFrame(median_imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Merge the imputed numeric data back with the non-numeric data
imputed_median_data = pd.concat([filtered_df[['Unnamed: 0']].reset_index(drop=True), imputed_median_data.reset_index(drop=True)], axis=1)

# Save the imputed dataframe to a new CSV file
imputed_median_data.to_csv('hb_median_simple.csv', index=False)

# Show the first few rows of the imputed dataframe
print(imputed_median_data.head())
