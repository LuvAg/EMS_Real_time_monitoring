"""Retrain simple models from data in data/ and save into model/.

This is a light-weight retraining script for development: it trains a
RandomForestRegressor for EV energy prediction (hourly) and another
RandomForest for satellite power estimation. The real project may want
CatBoost/PyTorch training pipelines; this script provides a fast path
for regenerating model artifacts when new data is appended via the dashboard.

Usage:
    python scripts\retrain_models.py

Outputs:
    model/ev_model.pkl
    model/sat_power_model.pkl

"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / 'data'
MODEL_DIR = BASE / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# EV model: predict Energy_kWh from Hour and Date features (simple)
ev_path = DATA_DIR / 'hourlyEVusage_cleaned.csv'
if ev_path.exists():
    df_ev = pd.read_csv(ev_path, parse_dates=['Datetime'])
    # Basic features
    df_ev['Hour'] = df_ev['Hour'].astype(int)
    df_ev['Day'] = df_ev['Datetime'].dt.day
    df_ev['Month'] = df_ev['Datetime'].dt.month
    X = df_ev[['Hour','Day','Month']].fillna(0)
    y = df_ev['Energy_kWh'].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    ev_model = RandomForestRegressor(n_estimators=50, random_state=42)
    ev_model.fit(X_train, y_train)
    joblib.dump(ev_model, MODEL_DIR / 'ev_model.pkl')
    print('Saved EV model to', MODEL_DIR / 'ev_model.pkl')
else:
    print('EV data not found at', ev_path)

# Satellite power model: predict 'power' from sensors
sat_path = DATA_DIR / 'satelite_hourly_data.csv'
if sat_path.exists():
    df_sat = pd.read_csv(sat_path)
    # Ensure necessary columns
    features = ['solar radiation','module_temp','wind direction','wind speed']
    for f in features:
        if f not in df_sat.columns:
            df_sat[f] = 0.0
    Xs = df_sat[features].fillna(0)
    ys = df_sat['power'].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.1, random_state=42)
    sat_model = RandomForestRegressor(n_estimators=50, random_state=42)
    sat_model.fit(X_train, y_train)
    joblib.dump(sat_model, MODEL_DIR / 'sat_power_model.pkl')
    print('Saved satellite power model to', MODEL_DIR / 'sat_power_model.pkl')
else:
    print('Satellite data not found at', sat_path)
