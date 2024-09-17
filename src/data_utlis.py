# src/data_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(), inplace=True)
    
    X = df.drop(columns='churn')
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_model(model, dataset_name):
    model_path = f'models/{dataset_name}_model.pkl'
    joblib.dump(model, model_path)

def load_model(dataset_name):
    model_path = f'models/{dataset_name}_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model for {dataset_name} not found.")

def save_scaler(scaler, dataset_name):
    scaler_path = f'models/{dataset_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)

def load_scaler(dataset_name):
    scaler_path = f'models/{dataset_name}_scaler.pkl'
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler for {dataset_name} not found.")
