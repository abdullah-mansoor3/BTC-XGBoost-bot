import streamlit as st
import pandas as pd
import json
import joblib
import ccxt
import numpy as np
from datetime import datetime, timedelta
from predict import get_info_and_predict, fetch_last_100_candles


# Helper function: Round the time to the nearest previous hour
def round_to_previous_hour(input_time):
    return input_time.replace(minute=0, second=0, microsecond=0)


# Streamlit Interface
st.title("BTC/USDT Next Candle Prediction")

# Modes: Live Prediction or Historical Prediction
mode = st.radio("Choose Prediction Mode", ["Live Prediction", "Historical Prediction"])

# File paths for the model and scaling parameters
model_filename = "xgb_model.pkl"
scaling_params_filename = "data/scaling_params.json"

if mode == "Historical Prediction":
    # User inputs: Date and time for prediction
    timestamp_input = st.date_input("Select Date")
    time_input = st.time_input("Select Time")
    input_timestamp = datetime.combine(timestamp_input, time_input)

    # Automatically round to the previous closest hour
    input_timestamp = round_to_previous_hour(input_timestamp)

    # Display the rounded timestamp
    st.write(f"Selected date and time (rounded to the previous hour): {input_timestamp}")

    # Fetch prediction
    st.write("Fetching prediction...")
    predicted_next_closing_price = get_info_and_predict(input_timestamp, model_filename, scaling_params_filename)

    # Display the predicted value
    st.write(f"Predicted next closing price: **{predicted_next_closing_price:.2f}**")

    # Fetch the last 100 candles to calculate the error (if actual data is available)
    candles = fetch_last_100_candles(input_timestamp)
    actual_close = candles["close"].iloc[-1]  # Actual closing price of the last candle

    # Calculate and display the error
    error = actual_close - predicted_next_closing_price
    st.write(f"Actual closing price: **{actual_close:.2f}**")
    st.write(f"Prediction error (Actual - Predicted): **{error:.2f}**")

elif mode == "Live Prediction":
    # Live Prediction: Current time rounded to the previous hour
    current_time = datetime.now()
    input_timestamp = round_to_previous_hour(current_time)

    st.write("Fetching prediction...")
    predicted_next_closing_price = get_info_and_predict(input_timestamp, model_filename, scaling_params_filename)

    st.write(f"Predicted next closing price: **{predicted_next_closing_price:.2f}**")

    # Display the current candle information
    st.write(f"Waiting for the current candle to close (Time: {input_timestamp + timedelta(hours=1)})...")

    # Check if the current candle has closed
    remaining_time = (input_timestamp + timedelta(hours=1)) - datetime.now()
    minutes, seconds = divmod(remaining_time.total_seconds(), 60)
    st.write(f"Time remaining for the current candle to close: {int(minutes):02d}:{int(seconds):02d}")

    # Once the candle closes, fetch and calculate
    if remaining_time.total_seconds() <= 0:

        # Fetch the actual data for the closed candle
        candles = fetch_last_100_candles(input_timestamp + timedelta(hours=1))
        actual_close = candles["close"].iloc[-1]

        # Display the error
        error = actual_close - predicted_next_closing_price
        st.write(f"Actual closing price: **{actual_close:.2f}**")
        st.write(f"Prediction error (Actual - Predicted): **{error:.2f}**")
    else:
        st.write("Waiting for the current candle to close...")
