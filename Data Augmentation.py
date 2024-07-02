import pandas as pd
import requests
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def convert_timestamps(df):
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df.drop(columns=['t'], inplace=True)
    return df

def calculate_daily_returns(df, prev_close=None):
    if prev_close is not None:
        df.loc[df.index[0], 'prev_close'] = prev_close
    else:
        df['prev_close'] = df['c'].shift(1)
    df['daily_return'] = (df['c'] - df['prev_close']) / df['prev_close']
    df['abs_daily_return'] = df['daily_return'].abs()
    return df

def fetch_hourly_data(symbol, start_date, end_date, api_key):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/hour/{start_date}/{end_date}?apiKey={api_key}"
    response = requests.get(url)
    hourly_data = pd.DataFrame(response.json()['results'])
    hourly_data = convert_timestamps(hourly_data)
    hourly_data = calculate_daily_returns(hourly_data)
    return hourly_data

# Load the dataset
data = pd.read_csv('Outlier Dataset.csv')
data['date'] = pd.to_datetime(data['date'])

# Filter rows where 'outlier' is 1
outliers = data[data['outlier'] == 1]

# Generate date ranges for each outlier
date_ranges = pd.DataFrame({
    "start_date": outliers['date'] - timedelta(days=3),
    "end_date": outliers['date'] + timedelta(days=3),
    "outlier_date": outliers['date']
})

api_key = 'my_api_key'
symbol = 'C:USDEUR'

# Plotting setup
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 40))
axs = axs.flatten()

# Process each date range
for idx, (ax, (_, row)) in enumerate(zip(axs, date_ranges.iterrows())):
    hourly_data = fetch_hourly_data(symbol, row['start_date'].date(), row['end_date'].date(), api_key)
    
    # Fill missing values
    hourly_data.fillna(method='ffill', inplace=True)  # forward fill

    # Normalize data
    scaler = StandardScaler()
    hourly_data[['v', 'vw', 'o', 'c', 'h', 'l']] = scaler.fit_transform(hourly_data[['v', 'vw', 'o', 'c', 'h', 'l']])

    # Prepare data for LSTM
    n_features = 1
    n_timesteps = 3  # Number of timesteps per sequence
    X = []
    y = []

    for i in range(n_timesteps, len(hourly_data)):
        X.append(hourly_data[['c']].iloc[i-n_timesteps:i].values)
        y.append(hourly_data['c'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    # Build and fit LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_timesteps, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=150, verbose=1)

    # Predict and generate synthetic data
    predicted = model.predict(X)
    noise = np.random.normal(0, 0.01, predicted.shape)
    synthetic_data = predicted + noise

    # Correlation calculation
    real_data = hourly_data['c'].iloc[n_timesteps:].values
    correlation, _ = pearsonr(real_data, synthetic_data.flatten())
    print(f'Packet {idx+1} - Correlation: {correlation}')

    # Find index of the outlier in the normalized data
    outlier_index = hourly_data.index[hourly_data['date'] == row['outlier_date']]
    
    # Plotting real vs synthetic data
    ax.plot(real_data, label='Real Data', color='black', linewidth=4)
    if not outlier_index.empty:
        ax.plot(outlier_index[0] - n_timesteps, real_data[outlier_index[0] - n_timesteps], 'ro', markersize=10, label='Outlier')  # Mark the outlier with a larger red dot
    ax.plot(synthetic_data.flatten(), label='Synthetic Data', color='green', linestyle='--', linewidth=4)
    ax.set_title(f'Outlier {idx + 1} - Correlation: {correlation:.2f}')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Normalized Price')
    ax.legend()
    ax.grid(True)

# Display all the plots
plt.tight_layout()
plt.show()