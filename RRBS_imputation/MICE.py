# MICE IMPUTATION USING MAX_ITER=10 (DEFAULT)

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

non_numeric_data = filtered_df[['Unnamed: 0']]
numeric_data = filtered_df.drop(columns=['Unnamed: 0'])

# Convert all columns in numeric_data to numeric, coercing errors
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# Applying MICE Imputation
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_numeric_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Merge the imputed numeric data back with the non-numeric data
imputed_data = pd.concat([non_numeric_data.reset_index(drop=True), imputed_numeric_data.reset_index(drop=True)], axis=1)

# Save the imputed dataframe to a new CSV file
#imputed_data.to_csv('filtered_processed_healthyvbenign_imputed_mice.csv', index=False)

# Show the first few rows of the imputed dataframe
print(imputed_data.head())


# MICE TO MONITOR CONVERGENCE AND PLOT CHANGE GRAPH

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt

# Function to monitor imputation convergence and plot the change
def monitor_imputation_convergence(data, max_iter):
    imputer = IterativeImputer(max_iter=max_iter, random_state=0)
    previous_data = np.zeros(data.shape)
    change_values = []
    for iteration in range(max_iter):
        imputed_data = imputer.fit_transform(data)
        change = np.linalg.norm(imputed_data - previous_data)
        change_values.append(change)
        print(f"Iteration {iteration + 1}, Change: {change:.4f}")
        if change < 1e-4:  # Threshold for convergence, can be adjusted
            break
        previous_data = imputed_data
    return pd.DataFrame(imputed_data, columns=data.columns), change_values

# Load your filtered_df (example data)
# filtered_df = pd.read_csv('your_filtered_data.csv')

# Select numeric columns for imputation
numeric_data = filtered_df.select_dtypes(include=[float, int])

# Monitor convergence and get change values
imputed_df, change_values = monitor_imputation_convergence(numeric_data, max_iter=20)

# Merge non-numeric data back
non_numeric_data = filtered_df[['Unnamed: 0']]
final_imputed_data = pd.concat([non_numeric_data.reset_index(drop=True), imputed_df.reset_index(drop=True)], axis=1)

# Save the final imputed dataframe to a new CSV file
#final_imputed_data.to_csv('filtered_processed_healthyvbenign_imputed_mice_corrected.csv', index=False)

# Show the first few rows of the final imputed dataframe
print(final_imputed_data.head())

# Plot the change values to visualize convergence
plt.plot(range(1, len(change_values) + 1), change_values, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Change')
plt.title('Convergence of Iterative Imputation')
plt.grid(True)
plt.show()
