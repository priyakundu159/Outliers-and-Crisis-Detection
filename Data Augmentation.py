import pandas as pd
import requests
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


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
})

api_key = 'beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq'
symbol = 'C:USDEUR'
all_hourly_data = pd.DataFrame()

# Fetch and process hourly data for each outlier
for index, row in date_ranges.iterrows():
    hourly_data = fetch_hourly_data(symbol, row['start_date'].date(), row['end_date'].date(), api_key)
    all_hourly_data = pd.concat([all_hourly_data, hourly_data], ignore_index=True)


# Fill missing values
all_hourly_data.fillna(method='ffill', inplace=True)  # forward fill

# Normalize all_hourly_data
scaler = StandardScaler()
all_hourly_data[['v', 'vw', 'o', 'c', 'h', 'l']] = scaler.fit_transform(all_hourly_data[['v', 'vw', 'o', 'c', 'h', 'l']])

# Prepare all_hourly_data for LSTM (assuming 'c' is what we want to predict)
n_features = 1
n_timesteps = 3  # Number of timesteps per sequence
X = []
y = []

for i in range(n_timesteps, len(all_hourly_data)):
    X.append(all_hourly_data[['c']].iloc[i-n_timesteps:i].values)
    y.append(all_hourly_data['c'].iloc[i])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_timesteps, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, verbose=1)

# Predict the next points
predicted = model.predict(X)

# Introduce small variations
noise = np.random.normal(0, 0.01, predicted.shape)
synthetic_data = predicted + noise

# Compare correlation
from scipy.stats import pearsonr

real_data = all_hourly_data['c'].iloc[n_timesteps:].values
correlation, _ = pearsonr(real_data, synthetic_data.flatten())
print(f'Correlation between real and synthetic data: {correlation}')


# Plotting real vs synthetic data
plt.figure(figsize=(14, 7))
plt.plot(real_data, label='Real Data', color='black', linewidth=2)
plt.plot(synthetic_data.flatten(), label='Synthetic Data', color='orange', linestyle='--', linewidth=2)
plt.title('Comparison of Real and Synthetic Data')
plt.xlabel('Time (hours)')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.show()