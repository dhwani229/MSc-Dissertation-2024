# mode impuation (manually)

# Calculate the mode for each column, excluding NaNs
# Note: mode() returns a DataFrame, we need to take the first mode value for each column
modes = filtered_df.mode().iloc[0]

# Impute missing values with the mode of each column
filtered_df_imputed_mode = filtered_df.fillna(modes)

# Save the imputed dataframe to a new CSV file
#filtered_df_imputed_mode.to_csv('filtered_processed_healthyvbenign_imputed_mode.csv', index=False)

# Show the first few rows of the imputed dataframe
print(filtered_df_imputed_mode.head())


# mode imputation (SingleImputer)

import pandas as pd
from sklearn.impute import SimpleImputer

# Assuming filtered_df is already available

# Select only the numeric columns
numeric_data = filtered_df.select_dtypes(include=[float, int])

# Applying Mode Imputation
mode_imputer = SimpleImputer(strategy='most_frequent')
imputed_mode_data = pd.DataFrame(mode_imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Merge the imputed numeric data back with the non-numeric data
imputed_mode_data = pd.concat([filtered_df[['Unnamed: 0']].reset_index(drop=True), imputed_mode_data.reset_index(drop=True)], axis=1)

# Save the imputed dataframe to a new CSV file
imputed_mode_data.to_csv('hb_mode_simple.csv', index=False)

# Show the first few rows of the imputed dataframe
print(imputed_mode_data.head())
