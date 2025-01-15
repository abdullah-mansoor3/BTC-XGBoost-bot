import joblib
import pandas as pd
import numpy as np
import json
import ccxt
import os
import pandas_ta as ta
import xgboost as xgb
import sklearn


def fetch_last_100_candles(candle_time, symbol='BTC/USDT', timeframe='1h'):
    """
    Fetches the last 100 candles for the given symbol and timeframe using the Binance API.

    Parameters:
    candle_time (datetime): The reference time to fetch candles before.
    symbol (str): The trading pair symbol (default 'BTC/USDT').
    timeframe (str): The timeframe for each candle (default '1h').

    Returns:
    pd.DataFrame: DataFrame containing the last 100 candles.
    """
    # Binance API credentials
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

    # Initialize Binance Exchange
    binance = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
        'rateLimit': 1200,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'  # Change to 'spot' for spot markets
        }
    })

    # Convert the input time to timestamp (milliseconds)
    candle_time_timestamp = int(candle_time.timestamp() * 1000)

    # Get the last 100 candles before the given time
    candles = binance.fetch_ohlcv(symbol, timeframe, since=candle_time_timestamp - (100 * 3600 * 1000), limit=100)
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

def calculate_features(df):
    """
    Calculate necessary features (EMA, RSI, Bollinger Bands, etc.) for the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing historical price data.

    Returns:
    pd.Series: A Series containing the calculated features for the last candle.
    """
    
    # 1. Last 100 candlesticks average between high and low
    df['avg_100_candles'] = (df['high'] + df['low']).rolling(window=100).mean()

    # 3. EMAs
    df['ema_20'] = ta.ema(df['close'], length=20)

    # 6. Bollinger Bands
    bollinger = ta.bbands(df['close'], length=20, std=2)
    df['bollinger_upper'] = bollinger['BBU_20_2.0']
    df['bollinger_lower'] = bollinger['BBL_20_2.0']

    # 7. Volume over last 100 candles
    df['volume_100'] = df['volume'].rolling(window=100).sum()

    # Only return the latest row (for the next prediction)
    return df.iloc[-1]

def load_scaling_params(scaling_params_filename):
    """
    Load scaling parameters from the given JSON file.

    Parameters:
    scaling_params_filename (str): Path to the scaling parameters file.

    Returns:
    dict: A dictionary containing scaling parameters.
    """
    with open(scaling_params_filename, 'r') as f:
        scaling_params = json.load(f)
    return scaling_params


def predict_next_closing_price(model_filename, input_data, scaling_params):
    """
    Function to predict the next closing price of a BTC/USDT candle ON HOURLY TIMEFRAME using the model and stored scaling parameters.

    Parameters:
    model_filename (str): Path to the saved model (.pkl).
    input_data (dict): Dictionary containing the input features for prediction.
    scaling_params (dict): Dictionary containing the scaling parameters for each feature.

    Returns:
    float: Predicted closing price of the next candle.
    """
    
    # Load the trained model
    model = joblib.load(model_filename)
    
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    
    # Standardize the input data using the stored mean and std from scaling_params
    for col in input_df.columns:
        if col in scaling_params:
            mean = scaling_params[col]['mean']
            std = scaling_params[col]['std']
            input_df[col] = (input_df[col] - mean) / std

    # Convert all columns to numeric (in case of non-numeric columns)
    input_df = input_df.apply(pd.to_numeric, errors='coerce')

    
    
    # Ensure no NaN values exist after conversion
    if input_df.isna().any().any():
        raise ValueError("Input data contains NaN values after conversion.")



    # Convert the DataFrame to a DMatrix object
    dmatrix_input = xgb.DMatrix(input_df)

    # Now, make the prediction
    predicted_value = model.predict(dmatrix_input)

    # Reverse the normalization to get the un-normalized value
    predicted_unscaled_value = (predicted_value[0] * scaling_params['next_close']['std']) + scaling_params['next_close']['mean']
    
    return predicted_unscaled_value

def get_info_and_predict(candle_time, model_filename, scaling_params_filename):
    """
    Function to fetch the last 100 candles, calculate features, and predict the next closing price.

    Parameters:
    candle_time (datetime): The timestamp for which to fetch the data.
    model_filename (str): Path to the saved model file.
    scaling_params_filename (str): Path to the scaling parameters file.

    Returns:
    float: Predicted next closing price.
    """
    
    # Fetch the last 100 candles
    candles = fetch_last_100_candles(candle_time)

    # Add the time-related features (hour, day_of_week, month)
    candles['day_of_week'] = candle_time.weekday()

    # Ensure 'timestamp' is removed from the DataFrame to avoid SettingWithCopyWarning
    updated_features = candles.drop(columns=['timestamp']).copy()
    
    # Calculate features for the last candle
    updated_features = calculate_features(candles)


    updated_features.drop('timestamp', inplace=True)


    # Load scaling parameters
    scaling_params = load_scaling_params(scaling_params_filename)

    updated_features = updated_features.drop("volume")

    # Predict the next closing price
    prediction = predict_next_closing_price(model_filename, updated_features, scaling_params)

    
    return prediction