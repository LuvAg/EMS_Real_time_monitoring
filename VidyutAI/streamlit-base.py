import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from model import LSTMRegressor, plotPrediction

st.title("Power Prediction for First 23 Rows")

DATA_PATH = r"F:\VidyutAI\data\satelite_hourly_data.csv"
MODEL_PATH = r"F:\VidyutAI\model\powerpredict.pkl"
SCALER_INPUT_PATH = r"F:\VidyutAI\model\scaler_input.pkl"
SCALER_POWER_PATH = r"F:\VidyutAI\model\scaler_power.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.subheader("Raw Data Preview")
st.dataframe(df.head(10))

if "power" not in df.columns:
    st.error("The dataset must contain a 'power' column.")
    st.stop()

if not (os.path.exists(SCALER_INPUT_PATH) and os.path.exists(SCALER_POWER_PATH)):
    st.error("Scaler files not found.")
    st.stop()

with open(SCALER_INPUT_PATH, "rb") as f:
    scaler_input = pickle.load(f)
with open(SCALER_POWER_PATH, "rb") as f:
    scaler_power = pickle.load(f)

temp_df = df.drop(columns=["power"], axis=1).astype(float)
temp_df_scaled = pd.DataFrame(
    scaler_input.transform(temp_df),
    columns=temp_df.columns
)
power_scaled = scaler_power.transform(df[["power"]])
p_temp_df = pd.concat([pd.DataFrame(power_scaled, columns=["power"]), temp_df_scaled], axis=1)

window_size = 23
subset_scaled = p_temp_df.iloc[100:window_size + 100, 1:].values

st.subheader("Subset (First 23 Rows Used for Prediction)")
st.dataframe(pd.DataFrame(subset_scaled, columns=temp_df.columns))

input_tensor = torch.tensor(subset_scaled, dtype=torch.float32).unsqueeze(0).to(device)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

plotPrediction(model, input_tensor.cpu().numpy(), power_scaled[123:147], scaler_input, scaler_power, device=device)
