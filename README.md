# README.md

## Churn Prediction System

### Overview

This project provides a churn prediction system using machine learning. It supports multiple datasets and offers model predictions and interpretability features using SHAP and LIME.


- **`data/`**: Directory for storing dataset files.
- **`src/`**: Contains the source code for data preprocessing, model training, and FastAPI endpoints.
- **`models/`**: Directory for storing saved machine learning models.

Usage
Start FastAPI Server:
Run the FastAPI server with:

uvicorn src.main:app --reload
The server will be accessible at http://127.0.0.1:8000.

API Endpoints:
Train Model:
POST http://127.0.0.1:8000/train/

Request Body:


{
  "dataset": "churn_dataset_1"
}

Predict Churn:
POST http://127.0.0.1:8000/predict/

Request Body:


{
  "dataset": "Telco-Customer-Churn",
  "features": [0, 1, 0, 10, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 100, 200]
}
Explain with SHAP:
POST http://127.0.0.1:8000/explain/shap/

Request Body:

{
  "dataset": "Telco-Customer-Churn",
  "features": [0, 1, 0, 10, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 100, 200]
}
Explain with LIME:
POST http://127.0.0.1:8000/explain/lime/

Request Body:


{
  "dataset": "Telco-Customer-Churn",
  "features": [0, 1, 0, 10, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 100, 200]
}
