import pandas as pd
import numpy as np

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and engineers features for the energy consumption dataset.
    - Combines Date and Time into Datetime
    - Sets Datetime as index
    - Converts columns to numeric
    - Interpolates missing values (requires DatetimeIndex)
    - Adds 'Other_consumption' feature
    - Removes negative values
    """
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df = df.drop(columns=['Date', 'Time'])
    if 'Datetime' in df.columns:
        df = df.set_index('Datetime')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("DataFrame index must be a DatetimeIndex for time interpolation. Please check your data format.")
    df = df.interpolate(method='time')
    if all(x in df.columns for x in ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']):
        df['Other_consumption'] = round(
            (df['Global_active_power'] * 1000 / 60) - (
                df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
            ), 1
        )
        df['Other_consumption'] = df['Other_consumption'].apply(lambda x: max(x, 0))
    return df

def create_xgboost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares features for XGBoost: hourly resample, time-based features, lag features.
    """
    df_hourly = df['Global_active_power'].resample('h').mean().interpolate(method='time')
    df_feat = df_hourly.to_frame().copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    for lag in range(1, 25):
        df_feat[f'lag_{lag}'] = df_feat['Global_active_power'].shift(lag)
    df_feat.dropna(inplace=True)
    return df_feat 