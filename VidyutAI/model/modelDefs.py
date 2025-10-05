from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class LSTMRegressor(nn.Module):
    def __init__(self, input_features=7, seq_len=23, hidden1=128, hidden2=64, fc=64, out_dim=24):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_features, hidden_size=hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.fc1 = nn.Linear(hidden2, fc)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(fc, out_dim)

    def forward(self, x):
        # x: (B, 23, 7)
        x, _ = self.lstm1(x)          # (B, 23, 128)
        x, _ = self.lstm2(x)          # (B, 23, 64)
        x = x[:, -1, :]               # (B, 64) - match Keras second LSTM with return_sequences=False
        x = self.act(self.fc1(x))     # (B, 64)
        x = self.fc2(x)               # (B, 24)
        return x

def plotPrediction(model, X, y, scaler, scaler_power, device='cpu'):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        pred = model(xb).cpu().numpy()[0]  # shape (24,)

    actuals = y  # shape (24,)

    # Inverse scale for power
    predictions_upscaled = scaler_power.inverse_transform(pred.reshape(-1,1)).reshape(-1)
    actuals_upscaled     = scaler_power.inverse_transform(actuals.reshape(-1,1)).reshape(-1)

    # Recover time features from input
    upscaled = scaler.inverse_transform(X[0][22].reshape(1, -1))
    day = int(np.ceil(upscaled[:, 0]))
    mo  = int(np.ceil(upscaled[:, 1]))
    hr  = int(np.ceil(upscaled[:, 2]))

    # Generate timestamps for 24 hours
    time_stamps = []
    hr_ = hr + 1
    for i in range(24):
        if hr_ > 17:
            hr_ = 6
        else:
            hr_ += 1
        time_stamps.append(hr_)

    # --- Plot in Streamlit ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'Predicted Power for Next 24 Hours (from HR:{hr}, Date:{day}/{mo})')
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Power (W)')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in time_stamps], rotation=45)
    ax.plot(predictions_upscaled, label='Predictions', ls=':', marker='o', color='blue')
    ax.plot(actuals_upscaled, label='Actuals', ls='-', marker='o', color='orange')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)