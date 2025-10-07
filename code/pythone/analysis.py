###
import os
import datetime
import pandas as pd
import numpy as np
import sys
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import sklearn.metrics as metrics
from sklearn.metrics import brier_score_loss, roc_curve, auc, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.metrics import roc_auc_score
prng = np.random.RandomState(20251006)
###

### Loading data
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '../../data/cleaned/train.csv')
test_path = os.path.join(base_dir, '../../data/cleaned/test.csv')
data = pd.read_csv(data_path)
test = pd.read_csv(test_path)
###

### Preparing data for modeling
y = data['is_popular']
X = data.drop(columns=['is_popular', 'article_id'])
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                              random_state=prng, stratify=y)

features_model1 = ['kw_avg_avg', 'LDA_03', 'kw_max_avg', 'LDA_02', 'data_channel_is_world']
features_model2 = features_model1 + ['average_token_length', 'num_imgs', 'num_videos',
                                     'kw_min_avg', 
                                     'data_channel_is_entertainment', 'data_channel_is_bus', 
                                     'data_channel_is_socmed', 'self_reference_avg_sharess', 
                                     'num_hrefs']

features_model3 = [col for col in X.columns]

print("Features used in Model 1:", features_model1)
print("Features used in Model 2:", features_model2)
print("Features used in Model 3:", features_model3)
###

### Model Training and Evaluation
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    return model, train_auc, test_auc
###

### Model 1 (few key features)
X_train_m1 = X_train_full[features_model1]
X_test_m1 = X_test_full[features_model1]
model1, train_auc1, test_auc1 = train_and_evaluate_model(X_train_m1, X_test_m1, y_train, y_test)
print("\nModel 1 (few key features):")
print("  Train ROC AUC:", train_auc1)
print("  Test ROC AUC:", test_auc1)
###

### Model 2 (more features)
X_train_m2 = X_train_full[features_model2]
X_test_m2 = X_test_full[features_model2]
model2, train_auc2, test_auc2 = train_and_evaluate_model(X_train_m2, X_test_m2, y_train, y_test)
print("\nModel 2 (more features):")
print("  Train ROC AUC:", train_auc2)
print("  Test ROC AUC:", test_auc2)
###

### Model 3 (all available predictive features)
X_train_m3 = X_train_full[features_model3]
X_test_m3 = X_test_full[features_model3]
model3, train_auc3, test_auc3 = train_and_evaluate_model(X_train_m3, X_test_m3, y_train, y_test)
print("\nModel 3 (all available predictive features):")
print("  Train ROC AUC:", train_auc3)
print("  Test ROC AUC:", test_auc3)
###
