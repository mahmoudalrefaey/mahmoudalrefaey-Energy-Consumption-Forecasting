import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_total_usage(df: pd.DataFrame):
    plt.figure(figsize=(15,5))
    df['Global_active_power'].plot(title='Total Active Power Usage (kW)')
    plt.show()

def plot_daily_energy_breakdown(df: pd.DataFrame):
    plt.figure(figsize=(15,6))
    df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Other_consumption']].resample('D').sum().plot(
        kind='area', stacked=True, title='Daily Energy Breakdown (Wh)')
    plt.show()

def plot_avg_hourly_usage(df: pd.DataFrame):
    plt.figure()
    df['hour'] = df.index.hour
    df.groupby('hour')['Global_active_power'].mean().plot(kind='bar', title='Avg Hourly Power Usage')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, title='Actual vs Predicted'):
    plt.figure(figsize=(14, 5))
    # Use modern soft color palette
    plt.plot(y_true.index, y_true, label='Actual', color='#1f77b4', linewidth=2, alpha=0.7)
    plt.plot(
        y_pred.index if hasattr(y_pred, 'index') else y_true.index,
        y_pred,
        label='Predicted',
        color='#ff7f0e',  # modern contrasting orange
        linewidth=2,
        alpha=0.9
    )

    # Modern styling
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time', fontsize=13)
    plt.ylabel('Power Consumption (kW)', fontsize=13)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.box(False)
    plt.show()