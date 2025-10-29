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
```

Install dependencies (recommend using a virtual environment):

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Download or place the UCI heart disease dataset (`heart.csv`) in the project directory.
2. Run the main script to train and evaluate models:

```bash
python main.py
```

3. The script will output performance metrics and save trained models in the `models/` directory.

4. To predict new samples, use the saved model as shown in the script or via a provided API (if implemented).

---

## 📊 Results

- **Best Model:** Random Forest achieved the highest ROC-AUC and overall accuracy.
- **Performance Metrics:**
    - Accuracy, Precision, Recall, F1, ROC-AUC reported for all models.
- **Feature Importance:** Top features identified include chest pain type, cholesterol, max heart rate, and age.

*(See detailed results in the outputs/logs or results section of the repo.)*

---

## ⚠️ Limitations

- **Small Dataset:** The UCI dataset is relatively small, which may limit generalizability.
- **Clinical Relevance:** Models are trained on public data and should not be used for clinical decisions.
- **Imbalanced Classes:** Even after SMOTE, some bias may remain.

---

## 🛠️ Future Work

- Integrate more datasets for improved generalization.
- Deploy as a web app or REST API for real-time predictions.
- Explore advanced models (XGBoost, Neural Networks).
- Add explainable AI features (e.g., SHAP, LIME).

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
