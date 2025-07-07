# Heart Disease Prediction 🫀

A machine learning project that predicts the presence of heart disease using clinical data from the UCI dataset.

---

## 🔗 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)

---

## 🧠 Overview

This project uses machine learning models (Logistic Regression and Random Forest) to predict the likelihood of heart disease in patients based on clinical attributes. It includes:
- Data preprocessing and outlier handling
- SMOTE for class imbalance
- Feature engineering
- Model training and hyperparameter tuning
- Evaluation using accuracy, precision, recall, ROC-AUC
- Saving and loading trained models

---

## 📂 Dataset

The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease) and includes 14 features such as:
- Age, sex, chest pain type, cholesterol, max heart rate, etc.
- Target variable: `1` (has disease), `0` (no disease)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/utkarsh884/Heart-disease-prediction.git
cd Heart-disease-prediction
