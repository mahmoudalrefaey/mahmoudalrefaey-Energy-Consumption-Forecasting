# Energy Consumption Forecasting (XGBoost Only)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GxZFGnT8bKamGGWVJLyagC4G71_3mdnD?usp=sharing)

A modern, interactive Streamlit app for forecasting household energy consumption using a pre-trained XGBoost model. Compare actual vs predicted usage, forecast future hours, and explore your data with live EDA visualizations.

---

## 🚀 Features
- **XGBoost Model**: Fast, accurate forecasting of energy consumption (no retraining required)
- **Forecast Future Hours**: Predict up to 168 hours ahead with rolling lag features
- **Actual vs Predicted**: Visual and tabular comparison of model performance
- **Live EDA**: Interactive plots for data exploration (no static images needed)
- **Upload Your Data**: Optionally upload new raw data for forecasting (must match expected columns)
- **Model Evaluation Visuals**: View static model evaluation images (if provided)
- **UCI Data Download**: By default, the app fetches the dataset directly from the UCI ML Repository (no local CSV needed)
- **Forecast Button & Loading Spinner**: Data is only loaded after clicking 'Forecast', with a clear 'Downloading data' spinner

---

## 📁 Project Structure
```
streamlit_app/
├── app.py                # Main Streamlit dashboard
├── requirements.txt      # All Python dependencies
├── README.md             # This documentation
├── utils/                # Modular utility scripts
│   ├── preprocessing.py  # Data cleaning & feature engineering
│   ├── xgboost_utils.py  # XGBoost split & prediction helpers
│   ├── metrics.py        # Regression metrics
│   ├── plotting.py       # EDA and result plotting
│   └── ...
├── assets/
│   └── model_eval/       # (Optional) Static model evaluation images
models/
└── xgboost_energy_forecast.pkl  # Pre-trained XGBoost model

data/
└── raw/individual_household_electric_power_consumption.csv  # (Optional, not used by default)
```

---

## ⚡ Quickstart

### 1. **Install Requirements**
```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. **Run the Dashboard**
```bash
streamlit run app.py
```

- The app will open in your browser at `http://localhost:8501` by default.

### 3. **Using the App**
- **Choose Data Source**: Use the UCI dataset (downloaded live) or upload your own (must match expected columns).
- **Forecast Future Hours**: Select how many hours to forecast in the sidebar.
- **Click 'Forecast'**: Data is only loaded and processed after you click the button. A spinner will show 'Downloading data, please wait...'.
- **Visualize Results**: See actual vs predicted, future forecasts, and EDA plots.
- **Model Evaluation Tab**: View static images from `assets/model_eval/` (optional).

---

## 🛠️ Customization
- **Model**: Replace `models/xgboost_energy_forecast.pkl` with your own XGBoost model (must match feature engineering pipeline).
- **Data**: Place your cleaned CSV in `data/raw/` or upload via the app (not required by default).
- **EDA/Plots**: Modify or extend `utils/plotting.py` for custom visualizations.
- **Model Evaluation Images**: Add PNGs to `assets/model_eval/` to display in the dashboard.

---

## 📝 Data Requirements
- The input CSV must have columns: `Date`, `Time`, `Global_active_power`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`, etc.
- The pipeline will automatically clean, interpolate, and engineer features for XGBoost.

---

## 📦 Dependencies
- Python 3.8+
- streamlit
- xgboost
- pandas
- numpy
- scikit-learn
- matplotlib
- ucimlrepo

(See `requirements.txt` for exact versions.)

---

## 🤝 Credits
- Data: [UCI Machine Learning Repository - Individual household electric power consumption Data Set](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- Dashboard: Built with [Streamlit](https://streamlit.io/) and [XGBoost](https://xgboost.readthedocs.io/)
- Notebook: [Colab Notebook for Data Preparation & Training](https://colab.research.google.com/drive/1GxZFGnT8bKamGGWVJLyagC4G71_3mdnD?usp=sharing)

---

## 📬 Support
For questions, issues, or feature requests, please open an issue or contact the maintainer. 