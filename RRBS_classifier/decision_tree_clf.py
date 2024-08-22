# DECISION TREE CLASSIFIER

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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


# Assign labels based on sample names
labels_train = [0 if sample in healthy_samples else 1 for sample in train_data.columns[1:]]
labels_val = [0 if sample in healthy_samples else 1 for sample in val_data.columns[1:]]
labels_test = [0 if sample in healthy_samples else 1 for sample in test_data.columns[1:]]

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
    'decisiontreeclassifier__max_depth': [3, 5, 7, 10],
    'decisiontreeclassifier__min_samples_split': [2, 5, 10],
    'decisiontreeclassifier__min_samples_leaf': [1, 2, 4]
}

# Create a pipeline

pipeline = Pipeline([
    ('selectkbest', SelectKBest(score_func=f_classif)),
    ('decisiontreeclassifier', DecisionTreeClassifier(random_state=42))
])

# Perform grid search to find the best combination of parameters
grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

# Get the results of the grid search
results = pd.DataFrame(grid_search.cv_results_)

# Filter and display only the relevant columns
results = results[['param_selectkbest__k', 'param_decisiontreeclassifier__max_depth',
                   'param_decisiontreeclassifier__min_samples_split', 'param_decisiontreeclassifier__min_samples_leaf',
                   'mean_test_score']]
results.columns = ['n_features', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'mean_auc']

# Sort by AUC in descending order
results = results.sort_values(by='mean_auc', ascending=False)

print(results)

# Finding optimal parameters
optimal_params = grid_search.best_params_

print(f"Optimal parameters: {optimal_params}")

# Feature selection using the optimal number of features
select_k_best = SelectKBest(score_func=f_classif, k=optimal_params['selectkbest__k'])
X_train_kbest = select_k_best.fit_transform(X_train, y_train)
X_test_kbest = select_k_best.transform(X_test)

# Train the Decision Tree classifier with the optimal parameters
classifier = DecisionTreeClassifier(max_depth=optimal_params['decisiontreeclassifier__max_depth'],
                                    min_samples_split=optimal_params['decisiontreeclassifier__min_samples_split'],
                                    min_samples_leaf=optimal_params['decisiontreeclassifier__min_samples_leaf'],
                                    random_state=42)
classifier.fit(X_train_kbest, y_train)

# Predictions
y_pred = classifier.predict(X_test_kbest)
y_pred_proba = classifier.predict_proba(X_test_kbest)[:, 1]

# Evaluation on Test Set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Healthy", "Benign"]) # replace benign with malignant in HM classifier

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
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Healthy", "Benign"])
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