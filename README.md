# cs431-project
CS 431 BADDIES

# COVID-19 Geospatial Forecasting Model

## Overview

The COVID-19 pandemic demonstrated the absolute necessity of having timely predictions regarding the rapid spread of infectious diseases. Traditional epidemiological models rely on predefined assumptions that may not capture the randomness of real-world disease transmission. By leveraging machine learning algorithms, this project aims to develop a predictive model that can geospatially forecast COVID-19 case trends. This model will be invaluable to researchers, healthcare professionals, and scientists by providing insights into locations at risk for future COVID-19 outbreaks—or potentially other infectious diseases.

## Problem Statement

Humans want to know what will happen next, and quickly. Predicting the who, what, when, where, and why of disease spread is essential for effective public health interventions. Our approach will move beyond traditional epidemiological models by incorporating machine learning to better capture the complex dynamics of disease spread.

## Tasks to Finish

- **Data Collection & Preprocessing:**
  - Acquire COVID-19 case data from the [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19) GitHub repository.
  - Load data into a Python notebook and perform Exploratory Data Analysis (EDA).
  - Handle missing values by either dropping or imputing them with appropriate strategies.

- **Data Visualization:**
  - Generate heatmaps to display correlations between variables.
  - Utilize hierarchical clustering to group similar data points and reveal underlying patterns.
  - Create line, scatter, bar plots, and histograms to analyze trends, relationships, and distributions.
  - Develop geospatial heatmaps to visualize regional trends and hotspots.

- **Feature Engineering:**
  - Merge data tables and engineer features based on insights from EDA.
  - Employ feature selection, encoding, scaling, and Principal Component Analysis (PCA) to improve model performance.

- **Model Development:**
  - Develop a custom Long Short-Term Memory (LSTM) model for time-series forecasting.
  - Fine-tune hyperparameters using Grid Search or Bayesian optimization.
  - Implement strategies to prevent overfitting and underfitting (e.g., cross-validation).

- **Model Evaluation:**
  - Split the dataset into training and evaluation sets.
  - Use evaluation metrics such as R-squared, adjusted R-squared, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to quantify model accuracy.

## Tech, Tools, and Systems

- **Programming Language:** Python
- **IDE:** Jupyter Notebooks
- **Version Control:** Git / GitHub

### Libraries & Tools

- **Pandas:** Data preprocessing and cleaning
- **NumPy:** Linear algebra and numerical computations
- **Matplotlib & Seaborn:** Data visualization
- **Scikit-Learn:** Machine learning and feature engineering
- **TensorFlow:** Deep learning (LSTM model)
- **GeoPandas:** Geospatial visualizations
- **Statsmodels:** Statistical analysis
- **SciPy:** Scientific computing

## Implementation & Evaluation Plan

### 1. Exploratory Data Analysis (EDA)
- **Data Understanding:** 
  - Analyze dataset shape, descriptive statistics, data types, and missing values.
  - Merge data from multiple CSV files, melting wide-format data into a time series format.
  - Remove duplicates for consistency.
- **Visualizations:**
  - Cluster maps, scatter plots, bar plots, histograms, and geospatial heatmaps.

### 2. Feature Engineering
- **Refining the Dataset:**
  - Feature selection to identify key predictors.
  - Feature encoding and scaling to enhance model interpretability.
  - Handling missing values via dropping, external data sources (e.g., FIPS data), or statistical imputation.
- **Dimensionality Reduction:**
  - Apply Principal Component Analysis (PCA) to highlight key patterns and reduce dimensionality.

### 3. Model Training
- **Model Development:**
  - Build an LSTM model to capture long-term dependencies in time-series data.
  - Optimize hyperparameters (number of LSTM units, learning rate, sequence length, dropout rate, batch size) using Grid Search or Random/Bayesian Search.

### 4. Performance Evaluation
- **Evaluation Strategy:**
  - Split the data into training and evaluation sets, and employ cross-validation.
  - Compare training and out-of-sample performance to avoid overfitting.
- **Metrics:**
  - Use R-squared, Adjusted R-squared, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to assess model performance.

## Timeline and Milestones

| Week         | Dates       | Tasks to Complete                                   | Team Member Responsibilities                            |
|--------------|-------------|-----------------------------------------------------|---------------------------------------------------------|
| **Week 1**   | 2/22-2/28   | Proposal Due                                        | - Kiana: Problem, Tasks to Finish<br>- Josh: Tech/Tools, Implementation<br>- Kyla: Implementation, Timeline  |
| **Week 1**   | 2/22-2/28   | Exploratory Data Analysis                           | - Kiana: Global dataset cleaning<br>- Josh: Start PCA<br>- Kyla: US dataset cleaning, visualizations  |
| **Week 2**   | 3/1-3/7     | Feature Engineering                                 | - Everyone: PCA, feature encoding/scaling, merging datasets |
| **Week 3**   | 3/8-3/14    | Model Training/Tuning                               | - Josh: Build initial model<br>- Kiana: Hyperparameter tuning via GridSearch/RandomSearch<br>- Kyla: Train multiple models and compare results |
| **Week 4**   | 3/15-3/21   | Model Performance Evaluation                        | - Everyone: Evaluate model performance, document results |
| **Week 5**   | 3/22-3/28   | SPRING BREAK                                        |                                                         |
| **Week 6**   | 3/29-4/4    | Begin Final Report and Slides                       | - Everyone: Gather results, create presentation visuals, outline report |
| **Week 7**   | 4/5-4/11    | Finalize Presentation                               | - Everyone: Finish slides, practice presentation, start final report |
| **Week 8**   | 4/12-4/18   | Final Presentation Due                              | - Everyone: Present and complete the final report       |

---

This README outlines the project's motivation, methodology, and schedule. The model is designed to be a robust forecasting tool that informs public health decisions, with potential applications beyond COVID-19 to other infectious diseases.

Feel free to adjust sections or add further details as the project progresses.
