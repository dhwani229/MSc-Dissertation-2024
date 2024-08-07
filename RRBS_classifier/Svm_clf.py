# CLASSIFIER USING SVM. SAME CODE APPLIED FOR ALL IMPUTED DATASETS BUT THIS EXAMPLE USES MEAN IMPUTED DATASET ONLY. 

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

os.chdir('/data/home/bt22912/files_needed/imputation/hb/mean') # This is where my imputed datasets were stored. 'mean' referred to where my mean imputed datasets were stored. this is modified according to which imputed dataset im using

# defining samples - benign samples for this eg. but malignant samples used in the HM classifier  
healthy_samples=[]
benign_samples=[]
malignant_samples=[]


# load data
train_data = pd.read_csv('train_hb_mean.csv', index_col=0)
val_data = pd.read_csv('val_hb_mean.csv', index_col=0)
test_data = pd.read_csv('test_hb_mean.csv', index_col=0)

# transpose data so samples are rows and features (DMPs) are columns
train_data = train_data.T
val_data = val_data.T
test_data = test_data.T

def assign_labels(samples):
    labels = []
    for sample in samples:
        if sample in healthy_samples:
            labels.append(0)  # Healthy
        elif sample in benign_samples:
            labels.append(1)  # Benign
        else:
            labels.append(-1)  # Unknown category
    return labels

labels_train = assign_labels(train_data.index)
labels_val = assign_labels(val_data.index)
labels_test = assign_labels(test_data.index)

def filter_samples(data, labels):
    mask = np.array(labels) != -1
    return data[mask], np.array(labels)[mask]

X_train, y_train = filter_samples(train_data.values, labels_train)
X_val, y_val = filter_samples(val_data.values, labels_val)
X_test, y_test = filter_samples(test_data.values, labels_test)


# Ensure target labels are categorical
y_train = pd.Series(y_train).astype(int)
y_val = pd.Series(y_val).astype(int)
y_test = pd.Series(y_test).astype(int)

# pipeline using SelectKBest and SVC
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('svm', SVC(kernel='linear', probability=True))
])

# parameter grid for GridSearchCV
param_grid = {
    'feature_selection__k': list(range(10, 30)),
    'svm__C': [0.1, 1, 10, 100]
}

# grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print(f"Optimal number of features: {best_params['feature_selection__k']}")
print(f"Optimal C: {best_params['svm__C']}")

# identify selected features
best_feature_selector = best_estimator.named_steps['feature_selection']
selected_feature_indices = best_feature_selector.get_support(indices=True)

feature_names = train_data.columns.values  # Column names (DMPs)

selected_feature_names = [feature_names[i] for i in selected_feature_indices]

print(f"Selected features ({len(selected_feature_names)}):")
for feature_name in selected_feature_names:
    print(feature_name)

# transform validation and test sets using the same steps as the training set
scaler = best_estimator.named_steps['scaler']
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_val_selected = best_feature_selector.transform(X_val_scaled)
X_test_selected = best_feature_selector.transform(X_test_scaled)

# Evaluate the model on the validation set
y_val_pred_proba = best_estimator.named_steps['svm'].predict_proba(X_val_selected)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUC: {val_auc}")

# Evaluate the model on the test set
y_test_pred = best_estimator.named_steps['svm'].predict(X_test_selected)
y_test_pred_proba = best_estimator.named_steps['svm'].predict_proba(X_test_selected)[:, 1]

accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred, target_names=["Healthy", "Benign"]) # LABEL CHANGE TO 'MALIGNANT' ON MALIGNANT CLASSIFIER 

tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp + tn) / (tp + tn + fp + fn)

# AUC and ROC Curve
auc = roc_auc_score(y_test, y_test_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

# Display the results
print(f"Accuracy: {accuracy: .4f}")
print(f"Sensitivity: {sensitivity: .4f}")
print(f"Specificity: {specificity: .4f}")
print(f"Accuracy: {acc}")
print(f"AUC: {auc}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# visualise 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Healthy", "Benign"])
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# FOR VALIDATION SET METRICS

# VALIDATION METRICS (FOR FINAL RESULTS TABLE)

# Predict labels and probabilities for the validation set
y_val_pred = best_estimator.named_steps['svm'].predict(X_val_selected)
y_val_pred_proba = best_estimator.named_steps['svm'].predict_proba(X_val_selected)[:, 1]

# METRICS
val_accuracy = accuracy_score(y_val, y_val_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
val_class_report = classification_report(y_val, y_val_pred, target_names=["Healthy", "Benign"])
val_auc = roc_auc_score(y_val, y_val_pred_proba)

val_tn, val_fp, val_fn, val_tp = val_conf_matrix.ravel()
val_sensitivity = val_tp / (val_tp + val_fn)
val_specificity = val_tn / (val_tn + val_fp)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Sensitivity: {val_sensitivity:.4f}")
print(f"Validation Specificity: {val_specificity:.4f}")
print(f"Validation AUC: {val_auc:.4f}")
print("Validation Confusion Matrix:")
print(val_conf_matrix)
print("Validation Classification Report:")
print(val_class_report)
