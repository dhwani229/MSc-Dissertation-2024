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

healthy_samples = ['SRR6350325',
'SRR6350326',
'SRR6350327',
'SRR6350328',
'SRR6350329',
'SRR6350330',
'SRR6350331',
'SRR6350332',
'SRR6350333',
'SRR6350334',
'SRR6350335',
'SRR6350336',
'SRR6350337',
'SRR6350338',
'SRR6350339',
'SRR6350340',
'SRR6350341',
'SRR6350342',
'SRR6350343',
'SRR6350344',
'GSM2877445',
'SRR6350347',
'SRR6350348',
'SRR6350349',
'SRR6350350',
'SRR6350351',
'SRR6350352',
'SRR6350353',
'SRR6350354',
'SRR6350355',
'SRR6350356',
'SRR6350357',
'SRR6350358',
'SRR6350359',
'SRR6350360',
'SRR6350361',
'SRR6350362',
'SRR6350363',
'SRR6350364',
'SRR6350365',
'SRR6350366',
'SRR6350367']

benign_samples = ['SRR6435624',
'SRR6435625',
'GSM2910007',
'SRR6435629',
'SRR6435630',
'SRR6435631',
'SRR6435632',
'SRR6435633',
'SRR6435634',
'SRR6435635',
'SRR6435636',
'SRR6350297',
'SRR6350298',
'SRR6350299',
'SRR6350300',
'SRR6350301',
'SRR6350302',
'SRR6350303',
'SRR6350304',
'SRR6350305',
'SRR6350306',
'SRR6350307',
'SRR6350308',
'SRR6350309',
'SRR6350310',
'SRR6350311',
'SRR6350312',
'SRR6350313',
'SRR6350314',
'SRR6350315',
'SRR6350316',
'SRR6350317',
'SRR6350318',
'SRR6350319',
'SRR6350320',
'SRR6350321',
'SRR6350322',
'SRR6350323',
'SRR6350324',
'SRR6350402',
'SRR6350403',
'SRR6350404',
'GSM2877501',
'SRR6350407',
'GSM2877503',
'SRR6350410',
'SRR6350411',
'SRR6350412',
'GSM2877507',
'GSM2877508',
'SRR6350418',
'SRR6350419',
'SRR6350420',
'SRR6350421',
'SRR6350422',
'SRR6350423',
'SRR6350424',
'SRR6350425',
'GSM2877517',
'SRR6350428',
'SRR6350429',
'SRR6350430',
'SRR6350431',
'SRR6350432']

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
