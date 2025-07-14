import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ucimlrepo import fetch_ucirepo
from utils.preprocessing import clean_and_engineer_features, create_xgboost_features
from utils.metrics import regression_metrics
from utils.xgboost_utils import xgb_time_split, xgb_predict
from utils.plotting import plot_total_usage, plot_daily_energy_breakdown, plot_avg_hourly_usage, plot_correlation_matrix, plot_actual_vs_predicted

# --- Robust path handling ---
def get_abs_path(*parts):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, *parts)
    if not os.path.exists(path):
        path = os.path.abspath(os.path.join(here, '..', *parts))
    return path

MODEL_XGB_PATH = get_abs_path('..', 'models', 'xgboost_energy_forecast.pkl')
ASSETS_PATH = get_abs_path('assets')
MODEL_EVAL_PATH = os.path.join(ASSETS_PATH, 'model_eval')

st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")
st.title("⚡ Energy Consumption Forecasting Dashboard")
st.caption("XGBoost for household energy forecasting. Upload new data or use the UCI dataset.")

with st.form(key="data_form"):
    data_choice = st.radio(
        "How would you like to forecast?",
        ("Use UCI sample data", "Upload my own data")
    )
    uploaded_file = None
    if data_choice == "Upload my own data":
        uploaded_file = st.file_uploader("Upload a CSV file (must match sample data features)", type=["csv"])
    submit = st.form_submit_button("Forecast")

st.sidebar.header("Forecast Options")
future_hours = st.sidebar.number_input(
    "Forecast how many future hours? (XGBoost)", min_value=1, max_value=168, value=24, step=1
)

# --- Data loading ---
def load_data():
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)
    df = individual_household_electric_power_consumption.data.features
    df = clean_and_engineer_features(df)
    return df

def load_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=True)
    df = clean_and_engineer_features(df)
    return df

def load_model(path, name):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Could not load {name} model. Make sure '{os.path.basename(path)}' exists in the models folder.\nError: {e}")
        st.stop()

xgb_model = load_model(MODEL_XGB_PATH, "XGBoost")

data_source = None
df = None
if submit:
    with st.spinner("Downloading data, please wait..."):
        if uploaded_file:
            data_source = 'User Uploaded Data'
            df = load_uploaded_data(uploaded_file)
        else:
            data_source = 'UCI Sample Dataset'
            df = load_data()

if df is not None:
    tabs = st.tabs(["Model Forecasts", "Model Evaluation", "EDA", "Data Preview"])

    # --- Model Forecasts Tab ---
    with tabs[0]:
        st.subheader("Model Forecasts & Evaluation")
        st.write(f"Data Source: **{data_source}**")
        # XGBoost
        st.markdown("#### XGBoost Forecast")
        df_xgb = create_xgboost_features(df)
        X_train, X_test, y_train, y_test = xgb_time_split(df_xgb)
        y_pred_xgb = xgb_predict(xgb_model, X_test)
        if len(y_test) == 0 or len(y_pred_xgb) == 0:
            st.warning("No data available for evaluation. Please check your data or upload a compatible file.")
        else:
            metrics_xgb = regression_metrics(y_test, y_pred_xgb)
            with st.expander("Show Metrics"):
                st.metric("MAE", f"{metrics_xgb['MAE']:.3f}")
                st.metric("RMSE", f"{metrics_xgb['RMSE']:.3f}")
                st.metric("R²", f"{metrics_xgb['R2']:.3f}")
            plot_actual_vs_predicted(y_test, y_pred_xgb, title='XGBoost: Actual vs Predicted')
            st.pyplot(plt.gcf())
            plt.clf()
            
        # --- Future Forecast (XGBoost) ---
        st.markdown("**Forecast Future Hours (XGBoost):**")
        # Roll forward using last available lags
        history_days = 7 * 24
        last_row = df_xgb.iloc[-1:].copy()
        future_preds = []
        current = last_row.copy()
        for i in range(future_hours):
            X_input = current.drop(columns=['Global_active_power'])
            y_pred = xgb_model.predict(X_input)[0]
            future_preds.append(y_pred)
            # Roll lags: shift all lag columns, insert new prediction as lag_1
            for lag in range(24, 1, -1):
                current[f'lag_{lag}'] = current[f'lag_{lag-1}'].values
            current['lag_1'] = y_pred
            # Update time features
            current.index = [current.index[0] + pd.Timedelta(hours=1)]
            current['hour'] = current.index[0].hour
            current['dayofweek'] = current.index[0].dayofweek
            current['month'] = current.index[0].month
        future_xgb_index = pd.date_range(start=last_row.index[0] + pd.Timedelta(hours=1), periods=future_hours, freq='h')
        fig3, ax3 = plt.subplots(figsize=(7,3))
        
        # Create a clean and soft-styled figure
    fig3, ax3 = plt.subplots(figsize=(14, 5))

    # Plot the last 7 days of historical data
    ax3.plot(
        df_xgb.index[-history_days:],
        df_xgb['Global_active_power'][-history_days:],
        label='History',
        color='#1f77b4',  # modern blue
        linewidth=2,
        alpha=0.7
    )

    # Plot future forecast
    ax3.plot(
        future_xgb_index,
        future_preds,
        label=f'Forecast (+{future_hours}h)',
        color='#ff7f0e',  # soft contrasting orange
        linewidth=2.5,
        alpha=0.9
    )

    # Format the plot
    ax3.set_title(f'XGBoost Forecast: Next {future_hours} Hours', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel("Power Consumption (kW)", fontsize=12)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    # Improve date formatting
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig3.autofmt_xdate()

    # Display plot in Streamlit
    st.pyplot(fig3)

    # Display forecast data table
    future_xgb_df = pd.DataFrame({
        'Datetime': future_xgb_index,
        'Forecasted Power (kW)': future_preds
    }).set_index('Datetime')

    st.dataframe(future_xgb_df.style.format("{:.2f}"))

    # --- Model Evaluation Tab ---
    with tabs[1]:
        st.subheader("Model Evaluation Visualizations")
        st.write("These are static images from model evaluation (kept in assets/model_eval).")
        if os.path.exists(MODEL_EVAL_PATH):
            eval_imgs = [f for f in os.listdir(MODEL_EVAL_PATH) if f.endswith('.png')]
            for img in eval_imgs:
                st.image(os.path.join(MODEL_EVAL_PATH, img), caption=img, use_column_width=True)
        else:
            st.info("No model evaluation images found in assets/model_eval.")

    # --- EDA Tab ---
    with tabs[2]:
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("These plots are generated live from the current data.")
        st.markdown("**Total Active Power Usage (kW):**")
        plot_total_usage(df)
        st.pyplot(plt.gcf())
        plt.clf()
        st.markdown("**Daily Energy Breakdown (Wh):**")
        plot_daily_energy_breakdown(df)
        st.pyplot(plt.gcf())
        plt.clf()
        st.markdown("**Average Hourly Power Usage:**")
        plot_avg_hourly_usage(df)
        st.pyplot(plt.gcf())
        plt.clf()
        st.markdown("**Correlation Matrix:**")
        plot_correlation_matrix(df)
        st.pyplot(plt.gcf())
        plt.clf()

    # --- Data Preview Tab ---
    with tabs[3]:
        st.subheader("Data Preview")
        st.write(f"Preview of the first 100 rows from: **{data_source}**")
        st.dataframe(df.head(100))

st.caption("Built with Streamlit | Model: XGBoost | Data: UCI Household Power Consumption") 