# Data Science 1 Project: Predicting Popularity of a Journal

## Project structure

```plaintext
project-root/
├── README.md
├── code
│   ├── jupyter_notebook
│   │   └── DC1.ipynb
│   └── pythone
│       ├── analysis.py
│       ├── cleaning.py
│       └── visualisation.py
├── data
│   ├── cleaned
│   │   ├── test.csv
│   │   └── train.csv
│   └── raw
│       ├── test.csv
│       └── train.csv
├── results
│   └── results.md
└── visualisation
    ├── correlation_matrix.png
    ├── data_channels_distribution.png
    ├── is_popular_distribution.png
    └── pairplot_selected_features.png
```
## Data

### Raw Data
The raw dataset contains information about online news articles, including structural, keyword-based, and sentiment-related features.  
Files are stored in the `data/raw/` folder as `train.csv` and `test.csv`.

### Data Cleaning
Data cleaning was performed using `cleaning.py`.  
The process included:
- Verifying and handling missing values (**none found**)
- Ensuring consistent data types
- Removing irrelevant or duplicate features

### Clean Data
Cleaned and processed datasets are stored in `data/cleaned/`.  
These files were used for model training and evaluation.

---

## Code

### Notebook
The Jupyter notebook `DC1.ipynb` (under `code/jupyter_notebook/`) contains the complete workflow:
- Data exploration and visualization  
- Feature analysis  
- Model training and evaluation  
- Performance comparison  

### Python
The Python scripts under `code/pythone/` modularize the workflow:
- **cleaning.py** – handles data preparation and cleaning  
- **analysis.py** – performs model training, evaluation, and metric reporting  
- **visualisation.py** – generates exploratory and model-related plots  

---

## Visualisations
All key visual outputs are stored in the `visualisation/` folder.  
These include:
- **correlation_matrix.png** – shows relationships between numerical features  
- **data_channels_distribution.png** – distribution of articles across channels  
- **is_popular_distribution.png** – target variable balance  
- **pairplot_selected_features.png** – feature interactions and relationships with `is_popular`  

---

## Results
Results and interpretations of model performance are presented in `results/results.md`. 
