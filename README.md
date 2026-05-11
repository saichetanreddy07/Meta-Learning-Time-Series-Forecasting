# Meta-Learning Time Series Forecasting System

## Overview

This project implements a Meta-Learning based Time Series Forecasting System capable of automatically selecting the best forecasting model for a given time series dataset. The system compares forecasting performance between ARIMA and Prophet models and trains a meta-model using extracted statistical characteristics of time series data.

The project uses multiple frequencies from the M4 Forecasting Dataset including:

* Monthly
* Weekly
* Yearly

The goal of the project is to automate forecasting model selection using machine learning and meta-features extracted from historical time series.

---

# Features

* Multi-frequency time series forecasting
* ARIMA forecasting model
* Prophet forecasting model
* Meta-feature extraction
* Random Forest based meta-learning model
* RMSE based model comparison
* Automatic best-model selection
* Visualization pipeline
* Forecast comparison plots
* Feature importance analysis

---

# Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Scikit-learn
* Statsmodels
* Prophet
* Matplotlib

---

# Dataset

## Dataset Used

M4 Forecasting Competition Dataset

The project uses multiple frequencies from the M4 dataset:

* Monthly
* Weekly
* Yearly

## Dataset Source

[https://www.mcompetitions.unic.ac.cy/the-dataset/](https://www.mcompetitions.unic.ac.cy/the-dataset/)

## Note

Raw dataset files are not included in this repository due to GitHub file size limitations.

Place the downloaded files inside:

```text
data/raw/train/
```

Required files:

```text
Monthly-train.csv
Weekly-train.csv
Yearly-train.csv
```

---

# Project Workflow

## Step 1: Load Dataset

The system loads multiple M4 datasets from different frequencies.

## Step 2: Preprocessing

The data is cleaned, indexed, and split into train and test sets.

## Step 3: Meta-Feature Extraction

Statistical features are extracted from each time series:

* Mean
* Variance
* Standard Deviation
* Trend
* Autocorrelation
* Length

## Step 4: Forecasting Models

Two forecasting models are trained:

* ARIMA
* Prophet

## Step 5: Model Evaluation

Forecasting performance is evaluated using RMSE.

## Step 6: Meta Dataset Creation

The extracted features and best-performing model labels are stored in a meta-dataset.

## Step 7: Meta-Learning

A Random Forest classifier predicts the best forecasting model for unseen datasets.

## Step 8: Visualization

The system generates:

* RMSE comparison plots
* Feature importance plots
* Forecast comparison plots
* Best model distribution graphs

---

# Project Structure

```text
Meta-Learning-Time-Series-Forecasting/
│
├── data/
│   ├── processed/
│   │   └── meta_dataset.csv
│   │
│   └── raw/
│       └── train/
│           ├── Monthly-train.csv
│           ├── Weekly-train.csv
│           └── Yearly-train.csv
│
├── results/
│   ├── plots/
│   │   ├── arima_vs_prophet.png
│   │   ├── best_model_distribution.png
│   │   ├── feature_importance.png
│   │   └── rmse_comparison.png
│   │
│   └── scripts/
│       ├── generate_plots.py
│       ├── plot_feature_importance.py
│       └── plot_predictions.py
│
├── src/
│   ├── data/
│   │   ├── load_m4.py
│   │   └── preprocess.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── features/
│   │   └── meta_features.py
│   │
│   ├── models/
│   │   ├── base_models.py
│   │   ├── meta_model.py
│   │   └── prophet_model.py
│   │
│   └── pipeline/
│       ├── create_meta_dataset.py
│       └── run_pipeline.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/saichetanreddy07/meta-learning-time-series-forecasting-system.git
```

## Navigate to Project

```bash
cd meta-learning-time-series-forecasting-system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
pandas>=1.5.0,<3.0.0
numpy>=1.23.0,<2.0.0
scikit-learn>=1.2.0,<2.0.0
statsmodels>=0.13.5,<0.15.0
matplotlib>=3.7.0,<4.0.0
prophet>=1.1.5,<2.0.0
```

---

# Running the Project

## Run Complete Pipeline

```bash
python src/pipeline/run_pipeline.py
```

The pipeline performs:

1. Dataset loading
2. Preprocessing
3. Feature extraction
4. Forecasting
5. RMSE evaluation
6. Meta dataset generation
7. Meta-model training
8. Plot generation

---

# Output Files

## Meta Dataset

```text
data/processed/meta_dataset.csv
```

## Generated Plots

```text
results/plots/
```

Generated visualizations include:

* Forecast comparison
* RMSE comparison
* Feature importance
* Best model distribution

---

# Models Used

## ARIMA

ARIMA is a statistical forecasting model that combines autoregression, differencing, and moving averages for time series prediction.

## Prophet

Prophet is a forecasting framework developed by Meta for handling trend and seasonality in time series data.

## Random Forest Meta-Model

A Random Forest classifier is used to predict which forecasting model performs best for a given time series.

---

# Evaluation Metric

## RMSE

Root Mean Squared Error (RMSE) is used to evaluate forecasting performance.

Lower RMSE indicates better forecasting accuracy.

---

# Results

The project demonstrates:

* Automatic forecasting model selection
* Meta-learning based forecasting automation
* Multi-frequency forecasting analysis
* Statistical feature-based forecasting decisions

The system identifies whether ARIMA or Prophet performs better for a given dataset based on extracted time series characteristics.

---

# Future Improvements

Possible future enhancements:

* Add LSTM forecasting
* Add XGBoost forecasting
* Use larger datasets
* Add hyperparameter tuning
* Deploy Streamlit frontend
* Add advanced meta-features
* Add ensemble forecasting

---

# References

1. The M4 Competition: Results, findings, conclusion and way forward
2. Forecasting at Scale
3. FFORMA: Feature-based Forecast Model Averaging
4. Meta-Learning: A Survey
5. Meta-learning how to forecast time series
6. Forecasting: Principles and Practice

---

# Author

Badvel Veera Sai Chetan Reddy

Computer Science (Artificial Intelligence)
Manipal Institute of Technology Bengaluru
