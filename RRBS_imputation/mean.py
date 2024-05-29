# mean imputation (manually)

means = filtered_df.mean()

filtered_df_imputed_mean = filtered_df.fillna(means)

# Save the imputed dataframe to a new CSV file
#filtered_df_imputed_mean.to_csv('filtered_processed_healthyvbenign_imputed_mean.csv', index=False)

# Show the first few rows of the imputed dataframe
print(filtered_df_imputed_mean.head())

# mean imputation (SingleImputer)


from sklearn.impute import SimpleImputer

# Assuming filtered_df is already available

# Select only the numeric columns
numeric_data = filtered_df.select_dtypes(include=[float, int])

# Applying Mean Imputation
mean_imputer = SimpleImputer(strategy='mean')
imputed_mean_data = pd.DataFrame(mean_imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Merge the imputed numeric data back with the non-numeric data
imputed_mean_data = pd.concat([filtered_df[['Unnamed: 0']].reset_index(drop=True), imputed_mean_data.reset_index(drop=True)], axis=1)

# Save the imputed dataframe to a new CSV file
#imputed_mean_data.to_csv('hb_simple.csv', index=False)

# Show the first few rows of the imputed dataframe
print(imputed_mean_data.head())
