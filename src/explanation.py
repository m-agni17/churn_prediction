# src/explanations.py

import shap
import lime
import lime.lime_tabular

def explain_with_shap(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return shap_values

def explain_with_lime(model, X_train, X_test, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=['Not Churn', 'Churn'], verbose=True, mode='classification')
    return explainer
