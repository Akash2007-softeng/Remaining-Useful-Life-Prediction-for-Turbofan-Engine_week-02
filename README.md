# Remaining-Useful-Life-Prediction-for-Turbofan-Engine_week-02
# Turbofan Engine Remaining Useful Life Prediction

## Week 2: Feature Engineering and Machine Learning

### Project

This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines using machine learning.

The project uses multivariate time-series engine data containing operational settings and sensor measurements. The objective is to estimate how many operational cycles remain before an engine reaches failure.

## Week 2 Objective

The objective of Week 2 was to transform the processed dataset into machine-learning-ready data, perform feature engineering and develop baseline regression models for RUL prediction.

## Week 2 Activities

### 1. Feature Engineering

The processed dataset from Week 1 was used for further development.

The following features were created:

* Rolling mean of sensor measurements
* Rolling standard deviation of sensor measurements
* Cycle ratio
* Selected sensor measurements
* Operational settings

Rolling features were calculated separately for each engine unit to capture changes in sensor behavior over operational cycles.

### 2. Machine Learning Models

Three regression models were developed:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Linear Regression was implemented as a baseline model. Decision Tree and Random Forest were used to capture nonlinear relationships between engine sensor information and RUL.

### 3. Model Evaluation

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Lower MAE and RMSE indicate lower prediction error, while a higher R² indicates better ability to explain the variation in the target.

## Results

The performance of the models was compared using the evaluation metrics.

The actual results obtained during execution are recorded in:

`results/model_comparison.csv`

Visual results are available in the `results/` directory.

### Results Files

* `feature_importance.png`
* `actual_vs_predicted.png`
* `model_comparison.png`
* `model_comparison.csv`

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Week 2 Learning Outcomes

During Week 2, I gained practical experience in:

* Feature engineering
* Time-series feature creation
* Regression
* Model training
* Train-test splitting
* Model evaluation
* MAE
* RMSE
* R² Score
* Feature importance
* Comparing machine learning models

## Project Structure

```text
turbofan-rul-prediction/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_loading.ipynb
│   ├── 02_data_quality_analysis.ipynb
│   ├── 03_rul_calculation.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_sensor_degradation_analysis.ipynb
│   ├── 06_correlation_analysis.ipynb
│   ├── 07_feature_engineering.ipynb
│   ├── 08_linear_regression.ipynb
│   ├── 09_decision_tree.ipynb
│   └── 10_random_forest.ipynb
│
├── results/
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   ├── model_comparison.png
│   └── model_comparison.csv
│
├── reports/
│   ├── week1_progress_report.pdf
│   └── week2_progress_report.pdf
│
├── README.md
└── requirements.txt
```

## Next Stage

The next stage of development will focus on improving the prediction performance using advanced machine learning techniques, hyperparameter tuning and model optimization.
