# Heart Disease Prediction

This project predicts heart disease using the UCI Heart Disease dataset with machine learning. It implements logistic regression and Random Forest models, handles class imbalance, and includes comprehensive exploratory data analysis (EDA).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)

## Project Overview
The goal is to predict the presence of heart disease (binary classification: 0 = no disease, 1 = disease) using features like age, cholesterol, and chest pain type. The project includes:
- Data preprocessing (encoding, scaling).
- EDA with visualizations (correlation heatmap, feature distributions).
- Model training with Logistic Regression and Random Forest.
- Class imbalance handling using SMOTE.
- Hyperparameter tuning and cross-validation.
- Evaluation with accuracy, precision, recall, F1-score, and ROC-AUC.

## Dataset
The dataset (`heart.csv`) is sourced from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). It contains:
- 303 instances.
- 14 features (e.g., `age`, `sex`, `chol`, `target`).
- Binary target: 0 (no heart disease), 1 (heart disease).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/utkarsh884/Heart-disease-prediction.git
   cd Heart-disease-prediction
