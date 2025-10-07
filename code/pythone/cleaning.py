### Importing
import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(__file__)
train_path = os.path.join(base_dir, '../../data/raw/train.csv')
test_path = os.path.join(base_dir, '../../data/raw/test.csv')

data = pd.read_csv(train_path)
test = pd.read_csv(test_path)
###

### Cleaning
to_filter=data.isna().sum()
to_filter[to_filter>0].sort_values(ascending = False)
data.fillna(data.median(), inplace=True)
test.fillna(test.median(), inplace=True)
###

### saving
cleaned_dir = os.path.join(base_dir, '../../data/cleaned')
os.makedirs(cleaned_dir, exist_ok=True)
data.to_csv(os.path.join(cleaned_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(cleaned_dir, 'test.csv'), index=False)
###