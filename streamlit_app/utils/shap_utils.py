import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

def get_shap_summary(model: xgb.XGBRegressor, X_test: pd.DataFrame):
    """
    Returns SHAP values and displays a summary plot for XGBoost model.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
    return shap_values 