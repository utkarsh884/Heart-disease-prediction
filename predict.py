import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = StandardScaler()  # Note: Scaler should be saved during training for consistency

# Example new data (replace with actual input)
new_data = pd.DataFrame({
    'age': [55], 'sex': [1], 'cp': [2], 'trestbps': [130], 'chol': [250],
    'fbs': [0], 'restecg': [1], 'thalach': [150], 'exang': [0], 'oldpeak': [1.5],
    'slope': [2], 'ca': [0], 'thal': [3]
})

# Preprocess new data (match notebook preprocessing)
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
new_data['age_chol'] = new_data['age'] * new_data['chol']
new_data = pd.get_dummies(new_data, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
new_data[numerical_cols + ['age_chol']] = scaler.fit_transform(new_data[numerical_cols + ['age_chol']])

# Predict
prediction = model.predict(new_data)
print("Prediction (0 = No Disease, 1 = Disease):", prediction[0])