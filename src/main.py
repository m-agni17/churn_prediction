from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from data_utlis import load_and_preprocess_data, load_model, load_scaler,save_scaler
from model import evaluate_model, train_and_save_model
from explanation import explain_with_shap, explain_with_lime

app = FastAPI()

models = {}
scalers = {}
feature_names = {}

class PredictRequest(BaseModel):
    dataset: str
    features: list

class TrainRequest(BaseModel):
    dataset: str

class ExplainRequest(BaseModel):
    dataset: str
    features: list

@app.post("/train/")
def train_model(request: TrainRequest):
    dataset = request.dataset
    try:
        file_path = f'data/{dataset}.csv'
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)
        model = train_and_save_model(X_train, y_train, dataset)
        save_scaler(scaler, dataset)
        
        models[dataset] = model
        scalers[dataset] = scaler
        feature_names[dataset] = X_train.columns.tolist()
        
        accuracy = evaluate_model(model, X_test, y_test)
        return {"message": f"Model trained and saved for {dataset}", "accuracy": accuracy}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

@app.post("/predict/")
def predict_churn(request: PredictRequest):
    dataset = request.dataset
    if dataset not in models or dataset not in scalers:
        raise HTTPException(status_code=404, detail="Model or scaler not found for the specified dataset")
    
    model = models[dataset]
    scaler = scalers[dataset]
    
    data = np.array(request.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    
    return {"prediction": int(prediction[0])}

@app.post("/explain/shap/")
def explain_shap(request: ExplainRequest):
    dataset = request.dataset
    if dataset not in models or dataset not in scalers:
        raise HTTPException(status_code=404, detail="Model or scaler not found for the specified dataset")
    
    model = models[dataset]
    scaler = scalers[dataset]
    
    # Assuming request.features are provided in a similar format as training features
    data = np.array(request.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    
    shap_values = explain_with_shap(model, data_scaled)
    
    return {"shap_values": shap_values.tolist()}

@app.post("/explain/lime/")
def explain_lime(request: ExplainRequest):
    dataset = request.dataset
    if dataset not in models or dataset not in scalers:
        raise HTTPException(status_code=404, detail="Model or scaler not found for the specified dataset")
    
    model = models[dataset]
    scaler = scalers[dataset]
    
    explainer = explain_with_lime(model, scaler, scaler, feature_names[dataset])
    data = np.array(request.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    
    explanation = explainer.explain_instance(data_scaled[0], model.predict_proba)
    
    return {"lime_explanation": explanation.as_list()}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}
