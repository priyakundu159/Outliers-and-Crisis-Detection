## Importing Necessary Libraries

from pandas.tseries.offsets import BDay
import requests
from datetime import timedelta
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense



## Identifying Outliers in last 12 months

def fetch_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching data:", response.status_code, response.text)
        return None
    data = response.json()
    if 'results' not in data:
        print("No 'results' key in response:", data)
        return None
    return data

def calculate_daily_returns(df, prev_close=None):
    if prev_close is not None:
        df.loc[df.index[0], 'prev_close'] = prev_close
    else:
        df['prev_close'] = df['c'].shift(1)
    df['daily_return'] = (df['c'] - df['prev_close']) / df['prev_close']
    df['abs_daily_return'] = df['daily_return'].abs()
    return df

def get_top_outliers(df, n=10):
    return df.nlargest(n, 'abs_daily_return')

def update_outliers_list(current_df, historical_outliers_df, real_time_outliers_df, n=10):
    if 'source' not in current_df.columns:
        current_df['source'] = 'real-time'
    combined_df = pd.concat([historical_outliers_df, current_df])
    updated_outliers_df = combined_df.nlargest(n, 'abs_daily_return')
    updated_historical_outliers_df = updated_outliers_df[updated_outliers_df['source'] == 'historical']
    updated_real_time_outliers_df = updated_outliers_df[updated_outliers_df['source'] == 'real-time']
    return updated_historical_outliers_df, updated_real_time_outliers_df

def convert_timestamps(df):
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df.drop(columns=['t'], inplace=True)
    return df

# API key and endpoints
api_key = 'beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq'
today = pd.Timestamp.now().date()
start_date = today - pd.DateOffset(years=1)
start_date_formatted = start_date.strftime('%Y-%m-%d')
end_date = today - pd.DateOffset(days=1)
end_date_formatted = end_date.strftime('%Y-%m-%d')
symbol = 'C:USDCHF'
historical_url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date_formatted}/{end_date_formatted}?adjusted=true&sort=asc&apiKey={api_key}'
real_time_url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{today}/{today}?adjusted=true&sort=asc&apiKey={api_key}'

# Fetch and process historical data
historical_data = fetch_data(historical_url)
if historical_data:
    historical_df = pd.DataFrame(historical_data['results'])
    historical_df = convert_timestamps(historical_df)
    historical_df = calculate_daily_returns(historical_df)
    historical_df['source'] = 'historical'
    historical_outliers_df = get_top_outliers(historical_df)
else:
    print("Failed to fetch or process historical data.")

# Fetch and process real-time data
real_time_data = fetch_data(real_time_url)
if real_time_data and 'results' in real_time_data:
    real_time_df = pd.DataFrame(real_time_data['results'])
    real_time_df = convert_timestamps(real_time_df)
    # Use the last close from historical data
    last_close = historical_df['c'].iloc[-1] if not historical_df.empty else None
    real_time_df = calculate_daily_returns(real_time_df, prev_close=last_close)
    real_time_df['source'] = 'real-time'
    updated_historical_outliers_df, updated_real_time_outliers_df = update_outliers_list(real_time_df, historical_outliers_df, pd.DataFrame())
    # Update historical data
    historical_df = pd.concat([historical_df.iloc[1:], real_time_df])  # Keep historical data rolling
else:
    print("No new data available or failed to fetch real-time data.")
    
# Combine data for Top 10 Outliers
full_outlier_df = pd.concat([updated_historical_outliers_df, updated_real_time_outliers_df])

sorted_outliers_data = full_outlier_df.sort_values(by="date")



## Fetching Hourly data for 3 days prior and post outlier days


# Convert dates in dataset to datetime objects
sorted_outliers_data['date'] = pd.to_datetime(sorted_outliers_data['date'])

date_ranges = pd.DataFrame({
    "start_date": sorted_outliers_data['date'] - BDay(3),
    "end_date": sorted_outliers_data['date'] + BDay(3),
    "outlier_date": sorted_outliers_data['date'],
})

def calculate_daily_returns(df, prev_close=None):
    if prev_close is not None:
        df.loc[df.index[0], 'prev_close'] = prev_close
    else:
        df['prev_close'] = df['c'].shift(1)
    df['returns'] = (df['c'] - df['prev_close']) / df['prev_close']
    return df

def fetch_hourly_data_chunk(symbol, start_date, end_date, api_key):
    formatted_start_date = start_date.strftime('%Y-%m-%d')
    formatted_end_date = end_date.strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/hour/{formatted_start_date}/{formatted_end_date}?apiKey={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None
    
    response_data = response.json()
    
    if 'results' not in response_data:
        print(f"No 'results' in response: {response_data}")
        return None

    df = pd.DataFrame(response_data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df.drop(columns=['t'], inplace=True)
    
    return df

def fetch_and_process_hourly_data(symbol, start_date, end_date, api_key):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    
    # Split the date range into smaller chunks
    chunk_size = 3  # Fetch data in 7-day chunks
    date_ranges = [(start_date + timedelta(days=i*chunk_size), 
                    min(end_date, start_date + timedelta(days=(i+1)*chunk_size - 1)))
                   for i in range((end_date - start_date).days // chunk_size + 1)]

    # print((end_date - start_date).days // chunk_size + 1)
    all_data = []

    for start, end in date_ranges:
        chunk_data = fetch_hourly_data_chunk(symbol, start, end, api_key)
        if chunk_data is not None:
            all_data.append(chunk_data)
    
    if not all_data:
        print("No data fetched")
        return None
    
    df = pd.concat(all_data)
    hourly_data = calculate_daily_returns(df)
    hourly_data.set_index('date', inplace=True)
    
    full_index = pd.date_range(start=start_date, end=end_date + timedelta(days=1), freq='H')
    hourly_data = hourly_data.reindex(full_index)
    
    hourly_data.reset_index(inplace=True)
    hourly_data.rename(columns={'index': 'date'}, inplace=True)
    
    return hourly_data

# Initialize empty DataFrame
all_data = pd.DataFrame()

# Initialize an outlier identifier starting from 1 or any specific number
outlier_id = 1

for index, row in date_ranges.iterrows():
    # Convert start_date, end_date, and outlier_date to Timestamp for consistent comparison
    start_date_ts = pd.Timestamp(row['start_date'])
    end_date_ts = pd.Timestamp(row['end_date']) + pd.Timedelta(days=1)  # Extend the end date by one additional day
    outlier_date_ts = pd.Timestamp(row['outlier_date'])
    
    # Get hourly data for the range including 3 days before and after the outlier
    hourly_data = fetch_and_process_hourly_data(symbol, start_date_ts, end_date_ts, api_key)
    # print(hourly_data)
    # Check if hourly_data is not None before processing
    if hourly_data is not None:
        # Assign the current outlier_id to the data
        hourly_data['outlier_id'] = outlier_id

        # Filter out weekdends
        hourly_data = hourly_data[~hourly_data['date'].dt.weekday.isin([5,6])]
        
        # prior_data from start_date to outlier_date inclusive
        prior_data = hourly_data[(hourly_data['date'] >= start_date_ts) & (hourly_data['date'] < outlier_date_ts)]
        prior_data["day type"] = "prior day"

        # outlier_data is for the hourly data on the day of the outlier
        outlier_data = hourly_data[(hourly_data['date'].dt.date == outlier_date_ts.date())]
        outlier_data["day type"] = "outlier day"
        
        # post_data from the day after outlier_date to end_date
        post_outlier_ts = outlier_date_ts + pd.Timedelta(days=1)  # Starting the day after the outlier_date
        post_data = hourly_data[(hourly_data['date'] > post_outlier_ts) & (hourly_data['date'] <= end_date_ts)]
        post_data["day type"] = "post day"
        # print(post_outlier_ts)
        # print(end_date_ts)
        # print(post_data)
        
        # Concatenate the data from this iteration to the cumulative DataFrame
        all_data = pd.concat([all_data, prior_data, outlier_data, post_data])

        # Increment the outlier_id for the next iteration
        outlier_id += 1

    else:
        print(f"Data not available for symbol {symbol} from {row['start_date'].date()} to {row['end_date'].date()}")

# Add the day column to the final DataFrame
all_data['day'] = all_data['date'].dt.day_name()
# Optionally, you can reset the index of the final DataFrame if it becomes non-unique after concatenations
all_data.reset_index(drop=True, inplace=True)



## Calculating and comparing similarity scores


def dtw_distance(series1, series2):
    n, m = len(series1), len(series2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(series1[i-1] - series2[j-1])
            # Take last min from a square sub-matrix
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
            
    return dtw_matrix[n, m]

# Load your data
data = pd.read_csv('USDCHF_hourly.csv')
# Fill null values with ffill then bfill to ensure all nulls are handled
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
all_data.fillna(method='ffill', inplace=True)
all_data.fillna(method='bfill', inplace=True)

# Filter Prior Day Data
prior_day_data = data[data['day type'] == 'prior day']

# # Main series to compare others against
main = all_data[all_data['day type'] == 'prior day']
ts_main_id = main[main['outlier_id'] == 10]['c'].values

# Calculate DTW distances between the current outlier's series and each of the other outliers
dtw_distances = {}
for id in prior_day_data["outlier_id"].unique():
    series = prior_day_data[prior_day_data['outlier_id'] == id]['c'].values
    distance = dtw_distance(ts_main_id, series)
    dtw_distances[id] = distance

# Sort the distances and get the top 10 lowest
top_10_ids = sorted(dtw_distances, key=dtw_distances.get)[:10]

lowest_dtw = 99999

# Output the results
for id in top_10_ids:
    print(f"DTW Distance between Current Outlier and ID {id}: {dtw_distances[id]:.2f}")
    if dtw_distances[id] < lowest_dtw:
        lowest_dtw = dtw_distances[id]
        best_outlier_match_id = id
    # best_outlier_match_id = min(best_outlier_match_id, dtw_distances[id])

# print(best_outlier_match_id)
data_aug = data[data['outlier_id'] == best_outlier_match_id]
# print(len(data_aug))

data_aug.drop(columns=["Unnamed: 0", "n", "prev_close", "returns", "outlier_id", "day"], inplace=True)
# Reset the index
data_aug.reset_index(drop=True, inplace=True)



## Augmenting Data for the choosen historical outlier


# Define the autoencoder model
def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)  # Use sigmoid if scaled [0,1]
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Prepare the data
features = ['v', 'vw', 'o', 'c', 'h', 'l']  # Specify the features to augment
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_aug[features])

# Train the autoencoder
autoencoder = create_autoencoder(data_scaled.shape[1])
autoencoder.fit(data_scaled, data_scaled, epochs=100, batch_size=32, shuffle=True, verbose=1)

# Generate synthetic data
synthetic_data = []
for _ in range(50):
    noise = np.random.normal(0, 0.1, data_scaled.shape)
    synthetic_data_scaled = autoencoder.predict(data_scaled + noise)
    synthetic_data.append(scaler.inverse_transform(synthetic_data_scaled))


# Initialize an outlier identifier starting from 1 or any specific number
outlier_id = 1

# Initialize an empty list to hold the individual synthetic dataframes
synthetic_dfs = []

# Iterate over each synthetic dataset
for i, synthetic in enumerate(synthetic_data):
    # Create a DataFrame from the synthetic data
    df = pd.DataFrame(synthetic, columns=features)
    # Add the 'day type' column from the existing 'data' DataFrame
    df['day type'] = data_aug['day type'].values  # Adjust as needed if 'day type' is not aligned
    # Add the outlier_id column
    df['outlier_id'] = outlier_id
    # Increment the outlier_id for the next iteration
    outlier_id += 1
    # Append the DataFrame to the list
    synthetic_dfs.append(df)

# Concatenate all the individual DataFrames into a single DataFrame
df_train = pd.concat(synthetic_dfs, ignore_index=True)

# Create a DataFrame from data_aug with outlier_id = 0
data_aug_df = data_aug[features + ['day type']].copy()
data_aug_df['outlier_id'] = 0

# Concatenate the synthetic DataFrames with the data_aug DataFrame
df_train = pd.concat([df_train, data_aug_df], ignore_index=True)

# Ensure the order of columns is consistent
df_train = df_train[features + ['outlier_id', 'day type']]




## Predictive Modeling


### Fetching 3 days prior and post data for the latest outlier for vailidating predictions

# Convert start_date, end_date, and outlier_date to Timestamp for consistent comparison
start_date_co = pd.Timestamp(date_ranges['start_date'].iloc[-1])
end_date_co = pd.Timestamp(date_ranges['end_date'].iloc[-1]) + pd.Timedelta(days=1)  # Extend the end date by one additional day
outlier_date_co = pd.Timestamp(date_ranges['outlier_date'].iloc[-1])

# Filter out weekdends
hourly_data = hourly_data[~hourly_data['date'].dt.weekday.isin([5,6])]

# prior_data from start_date to outlier_date inclusive
df_prior = hourly_data[(hourly_data['date'] >= start_date_co) & (hourly_data['date'] < outlier_date_co)]
df_prior["day type"] = "prior day"

# outlier_data is for the hourly data on the day of the outlier
df_outlier = hourly_data[(hourly_data['date'].dt.date == outlier_date_co.date())]
df_outlier["day type"] = "outlier day"

# post_data from the day after outlier_date to end_date
post_outlier_co = outlier_date_co + pd.Timedelta(days=1)  # Starting the day after the outlier_date
df_post = hourly_data[(hourly_data['date'] > post_outlier_co) & (hourly_data['date'] <= end_date_co)]
df_post["day type"] = "post day"


### Model training and prediction


# Function to prepare sequences for training
def prepare_sequences(df):
    sequences = []
    labels = []
    
    unique_ids = df['outlier_id'].unique()
    
    for oid in unique_ids:
        prior_data = df[(df['outlier_id'] == oid) & (df['day type'] == 'prior day')]['c'].values
        post_data = df[(df['outlier_id'] == oid) & (df['day type'] == 'post day')]['c'].values
        
        if len(prior_data) == 72 and len(post_data) >= 72:
            sequences.append(prior_data)
            labels.append(post_data[:72])
        else:
            print(f"Skipping outlier_id {oid} due to insufficient prior or post data")
    return np.array(sequences), np.array(labels)

def train_model(df_train):
	df_train.fillna(method='ffill', inplace=True)
	df_train.fillna(method='bfill', inplace=True)
	scaler = MinMaxScaler(feature_range=(0, 1))
	df_train['c'] = scaler.fit_transform(df_train[['c']])

	X_train, y_train = prepare_sequences(df_train)
	if X_train.size == 0:
		return None, None  # Early exit if no training data

	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

	model = Sequential([
		LSTM(50, return_sequences=True, input_shape=(72, 1)),
		LSTM(50),
		Dense(72)
	])
	model.compile(optimizer=Adam(learning_rate=0.05), loss='mean_squared_error')
	model.fit(X_train, y_train, epochs=1000, batch_size=64, verbose=2)
	return model, scaler

# Function to prepare a single sequence for prediction
def prepare_sequence_predict(df_prior, scaler):
	df_prior.drop(columns= ['prev_close', 'returns', 'n'], inplace=True)
	df_prior.fillna(method='ffill', inplace=True)
	df_prior.fillna(method='bfill', inplace=True)
	prior_data = df_prior['c'].values # "df_prior" should be the dataframe containing prior day's data for the current outlier.
	prior_data = scaler.transform(prior_data.reshape(-1, 1)).flatten()
	return prior_data.reshape(1, 72, 1)

# Function to predict using the model
def predict(model, df_prior, scaler):
    X_test = prepare_sequence_predict(df_prior, scaler)
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions.reshape(-1, 1))

# Function to validate predictions
def validate_predictions(predictions, df_post, scaler):
	df_post.drop(columns= ['prev_close', 'returns', 'n'], inplace=True)
	df_post.fillna(method='ffill', inplace=True)
	df_post.fillna(method='bfill', inplace=True)
	post_data = df_post['c'].values[:72] # "df_post" should be the dataframe containing post day's data for the current outlier.
	post_data_scaled = scaler.transform(post_data.reshape(-1, 1)).flatten()

	mse = mean_squared_error(post_data_scaled, predictions.flatten())
	mae = mean_absolute_error(post_data_scaled, predictions.flatten())
	rmse = np.sqrt(mse)
	r2 = r2_score(post_data_scaled, predictions.flatten())
	
	return mse, mae, rmse, r2

# Example usage:
model, scaler = train_model(df_train)
predictions = predict(model, df_prior, scaler)
mse, mae, rmse, r2 = validate_predictions(predictions, df_post, scaler)

# Now you can print these values in a different cell
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")