# RANDOM FOREST CLASSIFIER. SAME CODE FOR ALL SAMPLES BUT THIS E.G. USES MEAN

import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline

os.chdir('/data/home/bt22912/files_needed/imputation/hb/mean') # This is where my imputed datasets were stored. 'mean' referred to where my mean imputed datasets were stored. this is modified according to which imputed dataset im using

# defining samples - benign samples for this eg. but malignant samples used in the HM classifier 

healthy_samples=[]
benign_samples=[]
malignant_samples=[]

train_data = pd.read_csv('train_hb_mean.csv')
val_data = pd.read_csv('val_hb_mean.csv')
test_data = pd.read_csv('test_hb_mean.csv')

train_data.iloc[:, 1:] = train_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
val_data.iloc[:, 1:] = val_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
test_data.iloc[:, 1:] = test_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# assign labels based on sample names
def assign_labels(data, healthy_samples, benign_samples):
    labels = []
    for sample in data.columns[1:]:
        if sample in healthy_samples:
            labels.append(0)  # Healthy
        elif sample in benign_samples:
            labels.append(1)  # Benign
    return labels

labels_train = assign_labels(train_data, healthy_samples, benign_samples)
labels_val = assign_labels(val_data, healthy_samples, benign_samples)
labels_test = assign_labels(test_data, healthy_samples, benign_samples)

# extract features and target variables
X_train = train_data.iloc[:, 1:].T.values  
y_train = pd.Series(labels_train).astype(int)

X_val = val_data.iloc[:, 1:].T.values 
y_val = pd.Series(labels_val).astype(int)

X_test = test_data.iloc[:, 1:].T.values  
y_test = pd.Series(labels_test).astype(int)

# define a range of features to select and n_estimators for grid search
param_grid = {
    'selectkbest__k': list(range(10, 30)),
    'randomforestclassifier__n_estimators': [10, 50, 100, 150, 200, 250, 300]
}

# create a pipeline for feature selection and classification
pipeline = Pipeline([
    ('selectkbest', SelectKBest(score_func=f_classif)),
    ('randomforestclassifier', RandomForestClassifier(random_state=42))
])

# perform grid search to find the best combination of parameters using cross-validation
grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

# extract the best parameters
best_params = grid_search.best_params_

# print the best parameters
print("Best parameters: ", best_params)

# evaluate the best parameters on the validation set
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)

# get the mask of selected features
selected_features_mask = pipeline.named_steps['selectkbest'].get_support()

# extract feature names (chromosomal positions)
feature_names = train_data.iloc[:, 0].values

feature_names_array = np.array(feature_names)

# get the names of the selected features
selected_feature_names = feature_names_array[selected_features_mask]

print("Selected Feature Names:")
print(selected_feature_names)

# transform the validation set and predict
X_val_kbest = pipeline.named_steps['selectkbest'].transform(X_val)
y_val_pred = pipeline.named_steps['randomforestclassifier'].predict(X_val_kbest)
y_val_pred_proba = pipeline.named_steps['randomforestclassifier'].predict_proba(X_val_kbest)[:, 1]

# evaluation on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
val_class_report = classification_report(y_val, y_val_pred, target_names=["Healthy", "Benign"])

# Sensitivity, Specificity, and Accuracy for validation set
val_tn, val_fp, val_fn, val_tp = val_conf_matrix.ravel()
val_sensitivity = val_tp / (val_tp + val_fn)
val_specificity = val_tn / (val_tn + val_fp)
val_acc = (val_tp + val_tn) / (val_tp + val_tn + val_fp + val_fn)

# AUC for validation set
val_auc = roc_auc_score(y_val, y_val_pred_proba)

print("Validation Set Metrics:")
print(f"Accuracy: {val_accuracy}")
print(f"Sensitivity: {val_sensitivity}")
print(f"Specificity: {val_specificity}")
print(f"Accuracy: {val_acc}")
print(f"AUC: {val_auc}")
print("Confusion Matrix:")
print(val_conf_matrix)
print("Classification Report:")
print(val_class_report)

# transform the test set and predict
X_test_kbest = pipeline.named_steps['selectkbest'].transform(X_test)
y_pred = pipeline.named_steps['randomforestclassifier'].predict(X_test_kbest)
y_pred_proba = pipeline.named_steps['randomforestclassifier'].predict_proba(X_test_kbest)[:, 1]

# evaluation on the test set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Healthy", "Benign"])

# Sensitivity, Specificity, and Accuracy for test set
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp + tn) / (tp + tn + fp + fn)

# AUC for test set
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

print("\nTest Set Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Accuracy: {acc}")
print(f"AUC: {auc}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# visualise confusion matrix for test set
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Healthy", "Benign"])
cm_display.plot()
plt.title('Confusion Matrix - Test Set')
plt.show()

# Plot ROC Curve for test set
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Set')
plt.legend(loc='lower right')
plt.show()
