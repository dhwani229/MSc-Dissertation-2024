# ADABOOST CLASSIFIER

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import metrics

os.chdir('/data/home/bt22912/files_needed/imputation/hb/mean') # this e.g. uses mean imputed dataset. adjusted according to which imputation method used

# defining samples

healthy_samples=[]
benign_samples=[]
malignant_samples=[]

train_data = pd.read_csv('train_hb_mean.csv')
val_data = pd.read_csv('val_hb_mean.csv')
test_data = pd.read_csv('test_hb_mean.csv')

train_data.iloc[:, 1:] = train_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
val_data.iloc[:, 1:] = val_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
test_data.iloc[:, 1:] = test_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')


# Assign labels based on sample names
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

# Extract features and target variables
X_train = train_data.iloc[:, 1:].T.values  
y_train = pd.Series(labels_train).astype(int)

X_val = val_data.iloc[:, 1:].T.values 
y_val = pd.Series(labels_val).astype(int)

X_test = test_data.iloc[:, 1:].T.values  
y_test = pd.Series(labels_test).astype(int)

# Define the parameter grid for GridSearchCV
param_grid = {
    'selectkbest__k': list(range(10, 31)),
    'adaboostclassifier__n_estimators': [50, 100, 150, 200, 250, 300]
}

# Create a pipeline for feature selection and classification
pipeline = Pipeline([
    ('selectkbest', SelectKBest(score_func=f_classif)),
    ('adaboostclassifier', AdaBoostClassifier(random_state=42))
])

# Perform grid search to find the best combination of parameters
grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and their respective AUC
print("Best parameters found: ", grid_search.best_params_)
print("Best AUC on validation set: ", grid_search.best_score_)

# Train the final model using the optimal parameters on the training set
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(X_train, y_train)

# Transform the test set and predict
X_test_kbest = best_pipeline.named_steps['selectkbest'].transform(X_test)
y_pred = best_pipeline.named_steps['adaboostclassifier'].predict(X_test_kbest)
y_pred_proba = best_pipeline.named_steps['adaboostclassifier'].predict_proba(X_test_kbest)[:, 1]

# Evaluation on Test Set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Healthy", "Benign"])

# Sensitivity, Specificity, and Accuracy
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp + tn) / (tp + tn + fp + fn)

# AUC and ROC Curve
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Display the results
print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Accuracy: {acc}")
print(f"AUC: {auc}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualise confusion matrix

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Healthy", "Benign"])
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# VALIDATION METRICS 

X_val_kbest = best_pipeline.named_steps['selectkbest'].transform(X_val)
y_val_pred = best_pipeline.named_steps['adaboostclassifier'].predict(X_val_kbest)
y_val_pred_proba = best_pipeline.named_steps['adaboostclassifier'].predict_proba(X_val_kbest)[:, 1]


accuracy_val = accuracy_score(y_val, y_val_pred)
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
class_report_val = classification_report(y_val, y_val_pred, target_names=["Healthy", "Benign"])


tn_val, fp_val, fn_val, tp_val = conf_matrix_val.ravel()
sensitivity_val = tp_val / (tp_val + fn_val)
specificity_val = tn_val / (tn_val + fp_val)
auc_val = roc_auc_score(y_val, y_val_pred_proba)


print("VALIDATION SET METRICS:)
print(f"Accuracy: {accuracy_val}")
print(f"Sensitivity: {sensitivity_val}")
print(f" Specificity: {specificity_val}")
print(f" AUC: {auc_val}")
print("Confusion Matrix:")
print(conf_matrix_val)
print("Classification Report:")
print(class_report_val)
