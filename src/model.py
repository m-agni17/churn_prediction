# src/model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_utlis import save_model, save_scaler

def train_and_save_model(X_train, y_train, dataset_name):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    save_model(model, dataset_name)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
