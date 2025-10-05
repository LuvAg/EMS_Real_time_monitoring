import pandas as pd
import numpy as np
import pickle
import torch
import os
import joblib
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch for power prediction
try:
    import torch
    import torch.nn as nn
    POWER_MODEL_AVAILABLE = True
    
    # Define LSTM model class directly here to avoid import issues
    class LSTMRegressor(nn.Module):
        def __init__(self, input_features=7, seq_len=23, hidden1=128, hidden2=64, fc=64, out_dim=24):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size=input_features, hidden_size=hidden1, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
            self.fc1 = nn.Linear(hidden2, fc)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(fc, out_dim)

        def forward(self, x):
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x = x[:, -1, :]
            x = self.act(self.fc1(x))
            x = self.fc2(x)
            return x
            
except ImportError as e:
    POWER_MODEL_AVAILABLE = False
    print(f"PyTorch not available for power prediction: {e}")

class EnergyPredictionService:
    """
    Service for predicting both load and power generation using existing trained models
    """
    
    def __init__(self):
        self.base_path = os.path.dirname(__file__)
        self.model_path = os.path.join(self.base_path, "model")
        self.data_path = os.path.join(self.base_path, "data")
        
        # Load models and scalers
        self.load_prediction_model = None
        self.power_prediction_model = None
        self.load_scalers = None
        self.power_scalers = None
        
        # Load data
        self.load_data = None
        self.power_data = None
        
        # Device for PyTorch models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._load_load_prediction_model()
        self._load_power_prediction_model()
        self._load_historical_data()
    
    def _load_load_prediction_model(self):
        """Load the CatBoost load prediction model"""
        try:
            model_file = os.path.join(self.model_path, "catboost_ev_forecast.cbm")
            package_file = os.path.join(self.model_path, "catboost_package.pkl")
            
            if os.path.exists(model_file) and os.path.exists(package_file):
                self.load_prediction_model = CatBoostRegressor()
                self.load_prediction_model.load_model(model_file)
                
                with open(package_file, 'rb') as f:
                    self.load_scalers = joblib.load(f)
                
                print("Load prediction model loaded successfully")
            else:
                print("Load prediction model files not found")
                
        except Exception as e:
                print(f"Error loading load prediction model: {e}")
    
    def _load_power_prediction_model(self):
        """Load the PyTorch power prediction model"""
        try:
            if not POWER_MODEL_AVAILABLE:
                print("Power prediction model class not available")
                return
                
            model_file = os.path.join(self.model_path, "powerpredict.pkl")
            scaler_input_file = os.path.join(self.model_path, "scaler_input.pkl")
            scaler_power_file = os.path.join(self.model_path, "scaler_power.pkl")
            
            if all(os.path.exists(f) for f in [model_file, scaler_input_file, scaler_power_file]):
                # Load model
                with open(model_file, 'rb') as f:
                    self.power_prediction_model = pickle.load(f)
                self.power_prediction_model.to(self.device)
                self.power_prediction_model.eval()
                
                # Load scalers
                with open(scaler_input_file, 'rb') as f:
                    scaler_input = pickle.load(f)
                with open(scaler_power_file, 'rb') as f:
                    scaler_power = pickle.load(f)
                
                self.power_scalers = {
                    'input': scaler_input,
                    'power': scaler_power
                }
                
                print("Power prediction model loaded successfully")
            else:
                print("Power prediction model files not found")
                
        except Exception as e:
                print(f"Error loading power prediction model: {e}")
    
    def _load_historical_data(self):
        """Load historical data for predictions"""
        try:
            # Load EV usage data for load prediction
            ev_data_file = os.path.join(self.data_path, "hourlyEVusage_cleaned.csv")
            if os.path.exists(ev_data_file):
                self.load_data = pd.read_csv(ev_data_file)
                self.load_data['Datetime'] = pd.to_datetime(self.load_data['Datetime'])
                self.load_data = self.load_data.sort_values('Datetime').reset_index(drop=True)
                self.load_data.set_index('Datetime', inplace=True)
                
                # Prepare load data similar to loadPrediction.ipynb
                self._prepare_load_data()
                print("Load historical data loaded successfully")
            
            # Load satellite data for power prediction
            power_data_file = os.path.join(self.data_path, "satelite_hourly_data.csv")
            if os.path.exists(power_data_file):
                self.power_data = pd.read_csv(power_data_file)
                print("Power historical data loaded successfully")
                
        except Exception as e:
            print(f"Error loading historical data: {e}")
    
    def _prepare_load_data(self):
        """Prepare load data with features as in loadPrediction.ipynb"""
        if self.load_data is None:
            return
        
        try:
            # Fill missing hours
            complete_range = pd.date_range(
                start=self.load_data.index.min(), 
                end=self.load_data.index.max(), 
                freq='H'
            )
            self.load_data = self.load_data.reindex(complete_range, fill_value=0)
            self.load_data['Energy_kWh'] = self.load_data['Energy_kWh'].fillna(0)
            self.load_data['Load'] = self.load_data['Energy_kWh'].rolling(window=3, center=True).mean().fillna(self.load_data['Energy_kWh'])
            
            # Feature engineering
            self.load_data['Hour'] = self.load_data.index.hour
            self.load_data['DayOfWeek'] = self.load_data.index.dayofweek
            self.load_data['Month'] = self.load_data.index.month
            self.load_data['Day'] = self.load_data.index.day
            self.load_data['IsWeekend'] = self.load_data['DayOfWeek'].isin([5,6]).astype(int)
            self.load_data['IsBusinessHour'] = ((self.load_data['Hour'] >= 8) & (self.load_data['Hour'] <= 18)).astype(int)
            self.load_data['IsEvening'] = ((self.load_data['Hour'] >= 18) & (self.load_data['Hour'] <= 22)).astype(int)
            
            # Cyclic features
            self.load_data['Hour_sin'] = np.sin(2 * np.pi * self.load_data['Hour'] / 24)
            self.load_data['Hour_cos'] = np.cos(2 * np.pi * self.load_data['Hour'] / 24)
            self.load_data['DayOfWeek_sin'] = np.sin(2 * np.pi * self.load_data['DayOfWeek'] / 7)
            self.load_data['DayOfWeek_cos'] = np.cos(2 * np.pi * self.load_data['DayOfWeek'] / 7)
            self.load_data['Month_sin'] = np.sin(2 * np.pi * self.load_data['Month'] / 12)
            self.load_data['Month_cos'] = np.cos(2 * np.pi * self.load_data['Month'] / 12)
            
            # Lag features
            self.load_data['Lag_1h'] = self.load_data['Load'].shift(1)
            self.load_data['Lag_24h'] = self.load_data['Load'].shift(24)
            self.load_data['Lag_48h'] = self.load_data['Load'].shift(48)
            self.load_data['Lag_168h'] = self.load_data['Load'].shift(168)
            
            # Rolling features
            self.load_data['Rolling_mean_6h'] = self.load_data['Load'].rolling(6).mean()
            self.load_data['Rolling_mean_24h'] = self.load_data['Load'].rolling(24).mean()
            self.load_data['Rolling_std_24h'] = self.load_data['Load'].rolling(24).std()
            self.load_data['Rolling_max_24h'] = self.load_data['Load'].rolling(24).max()
            
            # Drop NaN values
            self.load_data = self.load_data.dropna().reset_index()
            
        except Exception as e:
            print(f"Error preparing load data: {e}")
    
    def predict_load_24h(self, station_id: Optional[str] = None) -> np.ndarray:
        """
        Predict load for the next 24 hours using CatBoost model
        Returns array of 24 hourly predictions in kWh
        """
        if self.load_prediction_model is None or self.load_scalers is None or self.load_data is None:
            print("Load prediction not available, using fallback")
            return self._fallback_load_prediction()
        
        try:
            # Sequence and static feature preparation
            history_len = 48
            
            feature_cols = [
                'Hour','DayOfWeek','Month','Day','IsWeekend','IsBusinessHour','IsEvening',
                'Hour_sin','Hour_cos','DayOfWeek_sin','DayOfWeek_cos','Month_sin','Month_cos',
                'Lag_1h','Lag_24h','Lag_48h','Lag_168h','Rolling_mean_6h','Rolling_mean_24h',
                'Rolling_std_24h','Rolling_max_24h'
            ]
            
            # Take the last sequence from data as input for prediction
            X_seq_input = self.load_data['Load'].values[-history_len:]
            X_static_input = self.load_data[feature_cols].iloc[-1].values.reshape(1, -1)
            
            # Scale inputs
            scaler_seq = self.load_scalers['scaler_seq']
            scaler_static = self.load_scalers['scaler_static']
            scaler_y = self.load_scalers['scaler_y']
            
            X_seq_scaled = scaler_seq.transform(X_seq_input.reshape(-1,1)).reshape(1, -1)
            X_static_scaled = scaler_static.transform(X_static_input)
            
            X_combined_input = np.concatenate([X_seq_scaled, X_static_scaled], axis=1)
            
            # Predict next 24 hours
            y_pred_scaled = self.load_prediction_model.predict(X_combined_input)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
            
            # Ensure positive values and reasonable range
            y_pred = np.maximum(y_pred, 0)
            
            return y_pred
            
        except Exception as e:
            print(f"Error in load prediction: {e}")
            return self._fallback_load_prediction()
    
    def predict_power_24h(self, station_id: Optional[str] = None) -> np.ndarray:
        """
        Predict solar power generation for the next 24 hours using PyTorch LSTM model
        Returns array of 24 hourly predictions in kW (scaled up from single panel)
        """
        if (self.power_prediction_model is None or 
            self.power_scalers is None or 
            self.power_data is None):
            print("Power prediction not available, using fallback")
            return self._fallback_power_prediction()
        
        try:
            # Prepare input similar to streamlit-base.py
            df = self.power_data.copy()
            
            if "power" not in df.columns:
                print("Power column not found in data")
                return self._fallback_power_prediction()
            
            # Scale the data
            temp_df = df.drop(columns=["power"], axis=1).astype(float)
            temp_df_scaled = pd.DataFrame(
                self.power_scalers['input'].transform(temp_df),
                columns=temp_df.columns
            )
            power_scaled = self.power_scalers['power'].transform(df[["power"]])
            
            # Use the last available window for prediction
            window_size = 23
            if len(temp_df_scaled) < window_size:
                print("Insufficient data for power prediction")
                return self._fallback_power_prediction()
            
            # Take the most recent data for prediction
            subset_scaled = temp_df_scaled.iloc[-window_size:].values
            
            # Create input tensor
            input_tensor = torch.tensor(subset_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.power_prediction_model.eval()
            with torch.no_grad():
                pred = self.power_prediction_model(input_tensor).cpu().numpy()[0]  # shape (24,)
            
            # Inverse scale for power
            predictions_upscaled = self.power_scalers['power'].inverse_transform(pred.reshape(-1,1)).reshape(-1)
            
            # Scale up from single panel (Wh) to grid scale (kW)
            # Convert Wh to kWh and multiply by scaling factor
            predictions_kw = (predictions_upscaled / 1000.0) * 300  # Scale factor of 300 as requested
            
            # Ensure positive values and apply realistic constraints
            predictions_kw = np.maximum(predictions_kw, 0)
            
            # Apply day/night cycle (solar generation should be zero at night)
            current_hour = datetime.now().hour
            for i in range(24):
                hour = (current_hour + i) % 24
                if hour < 6 or hour > 18:  # Night hours
                    predictions_kw[i] *= 0.1  # Very low generation at night
                elif hour < 8 or hour > 16:  # Dawn/dusk hours
                    predictions_kw[i] *= 0.5  # Reduced generation
            
            return predictions_kw
            
        except Exception as e:
            print(f"Error in power prediction: {e}")
            return self._fallback_power_prediction()
    
    def _fallback_load_prediction(self) -> np.ndarray:
        """Fallback load prediction based on typical patterns"""
        current_hour = datetime.now().hour
        base_load = 75.0  # Base load in kWh
        
        # Create realistic hourly pattern
        hourly_pattern = []
        for i in range(24):
            hour = (current_hour + i) % 24
            
            # Higher load during business hours and evening
            if 8 <= hour <= 18:  # Business hours
                multiplier = 1.5 + 0.3 * np.sin((hour - 8) * np.pi / 10)
            elif 18 <= hour <= 22:  # Evening peak
                multiplier = 1.8 + 0.2 * np.sin((hour - 18) * np.pi / 4)
            else:  # Night/early morning
                multiplier = 0.6 + 0.2 * np.sin(hour * np.pi / 12)
            
            # Add some randomness
            multiplier *= (1 + np.random.normal(0, 0.1))
            hourly_pattern.append(base_load * multiplier)
        
        return np.array(hourly_pattern)
    
    def _fallback_power_prediction(self) -> np.ndarray:
        """Fallback power prediction based on solar patterns"""
        current_hour = datetime.now().hour
        peak_generation = 60.0  # Peak generation in kW
        
        # Create realistic solar generation pattern
        generation_pattern = []
        for i in range(24):
            hour = (current_hour + i) % 24
            
            if 6 <= hour <= 18:  # Daylight hours
                # Bell curve for solar generation
                solar_angle = (hour - 12) / 6  # Normalize around noon
                generation = peak_generation * np.exp(-2 * solar_angle**2)  # Gaussian curve
                
                # Add some weather variability
                generation *= (0.7 + 0.3 * np.random.random())
            else:  # Night hours
                generation = 0.0
            
            generation_pattern.append(max(0, generation))
        
        return np.array(generation_pattern)
    
    def get_current_predictions(self, station_id: Optional[str] = None) -> Tuple[float, float]:
        """
        Get current hour predictions for load and generation
        Returns (load_kw, generation_kw)
        """
        try:
            load_24h = self.predict_load_24h(station_id)
            power_24h = self.predict_power_24h(station_id)
            
            # Return first hour (current) predictions
            return float(load_24h[0]), float(power_24h[0])
            
        except Exception as e:
            print(f"Error getting current predictions: {e}")
            # Fallback values
            return 75.0, 45.0
    
    def get_next_hour_predictions(self, station_id: Optional[str] = None) -> Tuple[float, float]:
        """
        Get next hour predictions for load and generation
        Returns (load_kw, generation_kw)
        """
        try:
            load_24h = self.predict_load_24h(station_id)
            power_24h = self.predict_power_24h(station_id)
            
            # Return second hour (next) predictions
            return float(load_24h[1]), float(power_24h[1])
            
        except Exception as e:
            print(f"Error getting next hour predictions: {e}")
            # Fallback values
            return 80.0, 50.0
    
    def is_available(self) -> dict:
        """Check which prediction services are available"""
        return {
            "load_prediction": self.load_prediction_model is not None,
            "power_prediction": self.power_prediction_model is not None,
            "load_data": self.load_data is not None,
            "power_data": self.power_data is not None
        }