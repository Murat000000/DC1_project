## Dataset Overview

The dataset contains no missing values, which allowed for direct use in modeling without imputation or cleaning. The target variable, **is_popular**, is binary and represents whether an article is among the most widely shared on social media. The mean value of *is_popular* is approximately **0.1216**, indicating that around **12.16%** of the articles in the dataset are considered popular. This reflects a degree of class imbalance that should be taken into account during model evaluation and selection.

---

## Exploratory Data Analysis

To gain an initial understanding of the dataset, some visualizations were done to explore feature distributions, correlations, and their relationships with the target variable *is_popular*.  

- The **correlation heatmap** shows strong positive correlations among groups of related variables, such as keyword-based metrics (`kw_min_min`, `kw_avg_avg`, etc.) and sentiment-based features (`rate_positive_words`, `avg_positive_polarity`, etc.), suggesting **multicollinearity** in parts of the dataset.  
- However, most variables show weak or no linear correlation with the target, indicating that article popularity is likely influenced by **complex, non-linear interactions** rather than by any single feature.  
- Some differences emerged between channels—for instance, *tech* and *entertainment* had a relatively higher proportion of popular articles compared to *lifestyle* or *socmed*—suggesting that the **content category** may be a moderately informative predictor.  

The **pairplot** of selected numeric features (`n_tokens_title`, `n_tokens_content`, `num_imgs`, `num_videos`, `average_token_length`, and `global_sentiment_polarity`) offered further insights: while no strong separation was observed between popular and non-popular articles, there were slight tendencies for popular articles to have more images or videos and a broader range in sentiment scores.  

Overall, the visual exploration highlights the **complexity** of the prediction task, with no dominant linear predictors, and supports the use of models capable of capturing subtle feature interactions.

---

## Modeling and Results

To evaluate the predictive potential of all models, we built three **logistic regression models** using increasingly rich sets of features based on their correlation with the target variable *is_popular*:

- **Model 1**: Constructed using the five most correlated features — `kw_avg_avg`, `LDA_03`, `kw_max_avg`, `LDA_02`, and `data_channel_is_world`.
- **Model 2**: Extended Model 1 by adding variables including structural features such as `average_token_length`, `num_imgs`, and `num_videos`.
- **Model 3**: Utilized **all available features** in the dataset.

### Performance Summary

| Model | Feature Set | Train ROC AUC | Test ROC AUC |
|:------|:-------------|:--------------|:--------------|
| Model 1 | Top 5 correlated features | 0.664 | 0.660 |
| Model 2 | Extended feature set | 0.686 | 0.680 |
| Model 3 | All features | **0.697** | **0.693** |

---

### Interpretation

The results show a clear trend: as more features were included, **model performance improved steadily**.  
A well-regularized linear model captures much of the predictive signal in the data and performs consistently **without significant overfitting**, even when using the full feature set.