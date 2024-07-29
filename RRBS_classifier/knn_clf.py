# CODE FOR KNN CLASSIFIER 

import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# change directory
os.chdir('/data/home/bt22912/files_needed/imputation/hb/mean') # changed dir to mean imputed files - replaced with all other imputation files but code remains the same

# defining samples - this one uses healthy and benign. in HM classifier, benign samples are replaced with malignant_samples

healthy_samples=[]
benign_samples=[]
malignant_samples=[]

train_data = pd.read_csv('train_hb_mean.csv', index_col=0)
val_data = pd.read_csv('val_hb_mean.csv', index_col=0)
test_data = pd.read_csv('test_hb_mean.csv', index_col=0)


train_data = train_data.apply(pd.to_numeric, errors='coerce')
val_data = val_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.apply(pd.to_numeric, errors='coerce')

def assign_labels(data, healthy_samples, benign_samples):
    labels = []
    for sample in data.columns:
        if sample in healthy_samples:
            labels.append(0)  # Healthy
        elif sample in benign_samples:
            labels.append(1)  # Benign
    return labels

labels_train = assign_labels(train_data, healthy_samples, benign_samples)
labels_val = assign_labels(val_data, healthy_samples, benign_samples)
labels_test = assign_labels(test_data, healthy_samples, benign_samples)

# extract features and target variables
X_train = train_data.values.T  # transpose the data
y_train = labels_train

X_val = val_data.values.T  # transpose the data
y_val = labels_val

X_test = test_data.values.T  # transpose the data
y_test = labels_test

# ensure target labels are categorical
y_train = pd.Series(y_train).astype(int)
y_val = pd.Series(y_val).astype(int)
y_test = pd.Series(y_test).astype(int)

# define the pipeline for feature selection and standardization
pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# define parameter grid for GridSearchCV
param_grid = {
    'feature_selection__k': list(range(10, 30)),
    'knn__n_neighbors': list(range(1, 31))
}

# perform grid search with cross-validation using training and validation sets
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print(f"Optimal number of features: {best_params['feature_selection__k']}")
print(f"Optimal number of neighbors: {best_params['knn__n_neighbors']}")

# Extract and print selected feature names
best_feature_selector = best_estimator.named_steps['feature_selection']
selected_feature_indices = best_feature_selector.get_support(indices=True)

# map the indices to the original feature names (DMPs)
feature_names = train_data.index.values  # DMPs row names 
selected_feature_names = [feature_names[i] for i in selected_feature_indices]

print(f"Selected features ({len(selected_feature_names)}):")
for feature_name in selected_feature_names:
    print(feature_name)

# cross-validation on the training data 
cross_val_auc = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validated AUC on training data: {np.mean(cross_val_auc)}")

# evaluate the model on the validation set
X_val_selected = best_feature_selector.transform(X_val)
X_val_scaled = best_estimator.named_steps['scaler'].transform(X_val_selected)
y_val_pred_proba = best_estimator.named_steps['knn'].predict_proba(X_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUC: {val_auc}")

# evaluate the model on the test set
X_test_selected = best_feature_selector.transform(X_test)
X_test_scaled = best_estimator.named_steps['scaler'].transform(X_test_selected)
y_test_pred = best_estimator.named_steps['knn'].predict(X_test_scaled)
y_test_pred_proba = best_estimator.named_steps['knn'].predict_proba(X_test_scaled)[:, 1]

# calculate the ROC curve and AUC for the test set
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = roc_auc_score(y_test, y_test_pred_proba)

# plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# confusion Matrix and Classification Report
y_pred = best_estimator.named_steps['knn'].predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Healthy", "Benign"])

# Sensitivity, Specificity, and Accuracy
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp + tn) / (tp + tn + fp + fn)

# display the results
print(f"Accuracy: {acc}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# extract the grid search results and display the table
results = grid_search.cv_results_
results_df = pd.DataFrame(results)
results_df = results_df[['param_feature_selection__k', 'param_knn__n_neighbors', 'mean_test_score', 'std_test_score', 'rank_test_score']]
results_df.columns = ['k', 'n_neighbors', 'mean_auc', 'std_auc', 'rank']
results_df = results_df.sort_values('rank')
print(results_df)

# VALIDATION METRICS (FOR FINAL RESULTS TABLE)

y_val_pred_proba = best_estimator.named_steps['knn'].predict_proba(X_val_scaled)[:, 1]
y_val_pred = best_estimator.named_steps['knn'].predict(X_val_scaled)

val_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUC: {val_auc}")


val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")


val_conf_matrix = confusion_matrix(y_val, y_val_pred)

tn, fp, fn, tp = val_conf_matrix.ravel()
val_sensitivity = tp / (tp + fn)
val_specificity = tn / (tn + fp)

print(f"Val Sensitivity: {val_sensitivity}")
print(f"Val Specificity: {val_specificity}")
print("Val Confusion Matrix:")
print(val_conf_matrix)
