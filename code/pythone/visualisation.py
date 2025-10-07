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
###

### Loading data
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '../../data/cleaned/train.csv')
test_path = os.path.join(base_dir, '../../data/cleaned/test.csv')
vis_dir = os.path.join(base_dir, '../../visualisation')
os.makedirs(vis_dir, exist_ok=True)
data = pd.read_csv(data_path)
test = pd.read_csv(test_path)
###

### Visualisation, e.g., distribution of target variable
plt.figure()
sns.countplot(x='is_popular', data=data)
plt.title('Distribution of is_popular')
plt.xlabel('is_popular')
plt.ylabel('Count')
plt.savefig(os.path.join(vis_dir, 'is_popular_distribution.png'))
plt.close()
###

### Visualisation, e.g., correlation matrix of numeric features
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
corr = data[numeric_cols].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix of Numeric Features')
plt.savefig(os.path.join(vis_dir, 'correlation_matrix.png'))
plt.close()
###

###
data_channels = ['data_channel_is_lifestyle', 'data_channel_is_entertainment',
                 'data_channel_is_bus', 'data_channel_is_socmed',
                 'data_channel_is_tech', 'data_channel_is_world']
# Melt the dataframe to plot counts per channel split by is_popular
df_channels = data.melt(id_vars=['is_popular'], value_vars=data_channels, 
                      var_name='data_channel', value_name='present')
# Keep only rows where the channel is active (value = 1)
df_channels = df_channels[df_channels['present'] == 1]

plt.figure()
sns.countplot(x='data_channel', hue='is_popular', data=df_channels)
plt.title('Data Channels Distribution by is_popular')
plt.xlabel('Data Channel')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig(os.path.join(vis_dir, 'data_channels_distribution.png'))
plt.close()

### Visualisation, e.g., pairplot of selected features
selected_features = ['n_tokens_title', 'n_tokens_content', 'num_imgs', 
                     'num_videos', 'average_token_length', 'global_sentiment_polarity', 
                     'is_popular']
sns.pairplot(data[selected_features], hue='is_popular', diag_kind='hist')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.savefig(os.path.join(vis_dir, 'pairplot_selected_features.png'))
plt.close()
###