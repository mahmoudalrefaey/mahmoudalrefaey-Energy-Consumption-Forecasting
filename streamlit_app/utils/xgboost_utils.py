import pandas as pd
import xgboost as xgb
from typing import Tuple

def xgb_time_split(df_feat: pd.DataFrame, target_col='Global_active_power', train_ratio=0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df_feat.drop(columns=target_col)
    y = df_feat[target_col]
    split = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test

def xgb_predict(model: xgb.XGBRegressor, X_test: pd.DataFrame) -> pd.Series:
    y_pred = model.predict(X_test)
    return pd.Series(y_pred, index=X_test.index) 