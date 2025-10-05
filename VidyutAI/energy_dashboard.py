import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from energy_management_system import EnergyManagementSystem, OperationMode
import random 
import json
try:
    import requests
except Exception:
    requests = None
try:
    # backend may or may not be available depending on how Streamlit is run
    import backend
except Exception:
    backend = None
import os

# Ensure a local ingest manager exists in the backend package when Streamlit runs
if backend is not None:
    try:
        if getattr(backend, 'manager', None) is None:
            try:
                from backend.ingest_manager import IngestManager as _IngestManager
                backend.manager = _IngestManager()
            except Exception:
                backend.manager = None
    except Exception:
        # keep going even if manager initialization fails
        pass

# Page configuration
st.set_page_config(
    page_title="Energy Management Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ems' not in st.session_state:
    st.session_state.ems = EnergyManagementSystem()
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None
if 'selected_station' not in st.session_state:
    st.session_state.selected_station = None

ems = st.session_state.ems

# If a local ingest manager is available, register EMS so it receives realtime updates
if backend is not None and getattr(backend, 'manager', None) is not None:
    try:
        ems.register_ingest_manager(backend.manager)
        st.sidebar.success("Local ingest manager active")
    except Exception:
        st.sidebar.warning("Failed to attach local ingest manager to EMS")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffcccc;
        color: #000000;
        border: 1px solid #ff6b6b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .alert-warning {
        background-color: #fff4cc;
        color: #000000;
        border: 1px solid #ffd93d;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .alert-maintenance {
        background-color: #ccffcc;
        color: #000000;
        border: 1px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .trading-positive {
        color: #4caf50;
        font-weight: bold;
    }
    .trading-negative {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚡ Smart Energy Management Dashboard")
st.markdown("---")

# Auto-refresh every 30 seconds
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
if auto_refresh:
    if datetime.now() - st.session_state.last_update > timedelta(seconds=30):
        ems.update_system_state()
        st.session_state.last_update = datetime.now()
        st.rerun()

# Manual refresh button
if st.sidebar.button("Refresh Now"):
    ems.update_system_state()
    st.session_state.last_update = datetime.now()
    st.rerun()

# Operation mode selector
st.sidebar.markdown("### Operation Mode")
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Automatic", "Manual"],
    index=0 if ems.operation_mode == OperationMode.AUTOMATIC else 1
)
ems.operation_mode = OperationMode.AUTOMATIC if mode == "Automatic" else OperationMode.MANUAL

if ems.operation_mode == OperationMode.AUTOMATIC:
    rl_enabled = st.sidebar.checkbox("Enable RL Algorithm", value=ems.rl_enabled)
    ems.rl_enabled = rl_enabled
    if not rl_enabled:
        st.sidebar.info("Using Rule-based Algorithm")

# Navigation
st.sidebar.markdown("### Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Market Overview", "Your Company", "Renewable Market", "Profit Forecast", "System Settings"]
)

# Current system status
st.sidebar.markdown("### System Status")
st.sidebar.metric("Grid Price", f"${ems.grid_price:.4f}/kWh")
st.sidebar.metric("Active Alerts", len(ems.maintenance_alerts))
st.sidebar.metric("Recent Trades", len([t for t in ems.trading_log if (datetime.now() - t['timestamp']).seconds < 3600]))

# Main content based on selected page
if page == "Market Overview":
    st.header("Market Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_generation = sum(
        sum(station.current_generation for station in company.stations)
        for company in ems.companies
    )
    total_load = sum(
        sum(station.current_load for station in company.stations)
        for company in ems.companies
    )
    total_surplus = sum(company.total_surplus for company in ems.companies)
    total_deficit = sum(company.total_deficit for company in ems.companies)
    
    with col1:
        st.metric("Total Generation", f"{total_generation:.1f} kW", f"{total_generation-total_load:.1f} kW")
    with col2:
        st.metric("Total Load", f"{total_load:.1f} kW")
    with col3:
        st.metric("Market Surplus", f"{total_surplus:.1f} kW")
    with col4:
        st.metric("Market Deficit", f"{total_deficit:.1f} kW")
    
    # Company comparison chart
    st.subheader("Company Energy Balance")
    
    companies_data = []
    for company in ems.companies:
        total_gen = sum(station.current_generation for station in company.stations)
        total_load_comp = sum(station.current_load for station in company.stations)
        companies_data.append({
            'Company': company.name,
            'Generation': total_gen,
            'Load': total_load_comp,
            'Balance': total_gen - total_load_comp,
            'Price': company.energy_price
        })
    
    df_companies = pd.DataFrame(companies_data)
    
    # Energy balance chart
    fig_balance = px.bar(df_companies, x='Company', y=['Generation', 'Load'],
                        title="Generation vs Load by Company",
                        barmode='group')
    st.plotly_chart(fig_balance, use_container_width=True)
    
    # Price comparison
    fig_price = px.bar(df_companies, x='Company', y='Price',
                      title="Energy Prices by Company",
                      color='Price',
                      color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Recent trading activity
    st.subheader("Recent Trading Activity")
    if ems.trading_log:
        recent_trades = sorted(ems.trading_log, key=lambda x: x['timestamp'], reverse=True)[:10]
        trades_df = pd.DataFrame(recent_trades)
        trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No recent trading activity")

elif page == "Your Company":
    st.header("Your Company Dashboard")
    
    # Company selector
    company_names = [company.name for company in ems.companies]
    selected_company_name = st.selectbox("Select Your Company", company_names)
    selected_company = next(c for c in ems.companies if c.name == selected_company_name)
    st.session_state.selected_company = selected_company
    
    # Company overview metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    company_generation = sum(station.current_generation for station in selected_company.stations)
    company_load = sum(station.current_load for station in selected_company.stations)
    
    with col1:
        st.metric("Total Generation", f"{company_generation:.1f} kW")
    with col2:
        st.metric("Total Load", f"{company_load:.1f} kW")
    with col3:
        balance = company_generation - company_load
        st.metric("Energy Balance", f"{balance:.1f} kW", 
                 delta_color="normal" if balance >= 0 else "inverse")
    with col4:
        st.metric("Energy Price", f"${selected_company.energy_price:.4f}/kWh")
    with col5:
        profit_color = "normal" if selected_company.current_hour_profit >= 0 else "inverse"
        st.metric("Current Hour Profit", f"${selected_company.current_hour_profit:.2f}", 
                 delta_color=profit_color)
    with col6:
        next_profit_color = "normal" if selected_company.expected_next_hour_profit >= 0 else "inverse"
        st.metric("Expected Next Hour", f"${selected_company.expected_next_hour_profit:.2f}", 
                 delta_color=next_profit_color)
    
    # Additional profit metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        cumulative_color = "normal" if selected_company.cumulative_profit >= 0 else "inverse"
        st.metric("Cumulative Profit", f"${selected_company.cumulative_profit:.2f}", 
                 delta_color=cumulative_color)
    with col2:
        if len(selected_company.profit_history) > 1:
            profit_trend = selected_company.profit_history[-1] - selected_company.profit_history[-2]
            trend_color = "normal" if profit_trend >= 0 else "inverse"
            st.metric("Profit Trend", f"${profit_trend:.2f}", 
                     delta_color=trend_color)
        else:
            st.metric("Profit Trend", "N/A")
    with col3:
        if selected_company.profit_history:
            avg_profit = np.mean(selected_company.profit_history)
            st.metric("Average Hourly Profit", f"${avg_profit:.2f}")
        else:
            st.metric("Average Hourly Profit", "N/A")
    
    # EV Charging & Battery Status Overview
    st.subheader("EV Charging & Battery Storage Overview")
    battery_status = ems.get_company_battery_status(selected_company)
    
    # Calculate EV charging metrics
    total_ev_load = company_load * 0.8  # Assume 80% of load is EV charging
    battery_supporting_evs = max(0, battery_status['total_discharging_power_kw'])
    solar_supporting_evs = min(company_generation, total_ev_load)
    grid_supporting_evs = max(0, total_ev_load - solar_supporting_evs - battery_supporting_evs)
    
    # EV Charging Status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("EV Charging Load", f"{total_ev_load:.1f} kW", help="Estimated EV charging demand (80% of total load)")
    with col2:
        st.metric("Battery → EVs", f"{battery_supporting_evs:.1f} kW", 
                 delta_color="normal", help="Power from batteries to charge EVs")
    with col3:
        st.metric("Solar → EVs", f"{solar_supporting_evs:.1f} kW", 
                 delta_color="normal", help="Solar power directly charging EVs")
    with col4:
        st.metric("Grid → EVs", f"{grid_supporting_evs:.1f} kW", 
                 delta_color="inverse" if grid_supporting_evs > 0 else "normal", 
                 help="Grid power needed for EV charging")
    
    # Battery Storage Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Capacity", f"{battery_status['total_capacity_kwh']:.1f} kWh")
    with col2:
        st.metric("Stored Energy", f"{battery_status['total_stored_kwh']:.1f} kWh")
    with col3:
        soc_color = "normal" if battery_status['total_soc_percent'] > 50 else "inverse"
        st.metric("Average SOC", f"{battery_status['total_soc_percent']:.1f}%", delta_color=soc_color)
    with col4:
        if battery_status['total_charging_power_kw'] > 0:
            st.metric("Charging", f"{battery_status['total_charging_power_kw']:.1f} kW", delta_color="normal")
        else:
            st.metric("Charging", "0.0 kW")
    with col5:
        if battery_status['total_discharging_power_kw'] > 0:
            st.metric("Discharging", f"{battery_status['total_discharging_power_kw']:.1f} kW", delta_color="inverse")
        else:
            st.metric("Discharging", "0.0 kW")
    
    # Battery visualization per station
    st.subheader("Battery Status by Station")
    
    station_names = []
    station_capacities = []
    station_stored = []
    station_socs = []
    
    for station in selected_company.stations:
        station_battery_data = battery_status['stations'][station.id]
        station_names.append(station.name)
        station_capacities.append(station_battery_data['total_capacity'])
        station_stored.append(station_battery_data['total_stored'])
        station_socs.append((station_battery_data['total_stored'] / station_battery_data['total_capacity'] * 100) if station_battery_data['total_capacity'] > 0 else 0)
    
    fig_battery = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Battery Capacity vs Stored Energy', 'Station Battery SOC', 'EV Charging Power Sources'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Capacity vs Stored chart
    fig_battery.add_trace(
        go.Bar(name='Total Capacity', x=station_names, y=station_capacities, 
               marker_color='lightblue', opacity=0.7),
        row=1, col=1
    )
    fig_battery.add_trace(
        go.Bar(name='Stored Energy', x=station_names, y=station_stored, 
               marker_color='darkblue'),
        row=1, col=1
    )
    
    # SOC percentage chart
    colors = ['red' if soc < 20 else 'orange' if soc < 50 else 'green' for soc in station_socs]
    fig_battery.add_trace(
        go.Bar(name='SOC %', x=station_names, y=station_socs, 
               marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # EV Charging Power Sources chart
    station_ev_loads = []
    station_battery_support = []
    station_solar_support = []
    station_grid_support = []
    
    for station in selected_company.stations:
        station_ev_load = station.current_load * 0.8  # 80% EV charging
        station_battery_power = max(0, sum(-battery.current_power for battery in station.batteries if battery.current_power < 0))
        station_solar_power = min(station.current_generation, station_ev_load)
        station_grid_power = max(0, station_ev_load - station_solar_power - station_battery_power)
        
        station_ev_loads.append(station_ev_load)
        station_battery_support.append(station_battery_power)
        station_solar_support.append(station_solar_power)
        station_grid_support.append(station_grid_power)
    
    fig_battery.add_trace(
        go.Bar(name='Battery → EV', x=station_names, y=station_battery_support, 
               marker_color='blue'),
        row=3, col=1
    )
    fig_battery.add_trace(
        go.Bar(name='Solar → EV', x=station_names, y=station_solar_support, 
               marker_color='orange'),
        row=3, col=1
    )
    fig_battery.add_trace(
        go.Bar(name='Grid → EV', x=station_names, y=station_grid_support, 
               marker_color='red'),
        row=3, col=1
    )
    
    fig_battery.update_xaxes(title_text="Stations", row=3, col=1)
    fig_battery.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig_battery.update_yaxes(title_text="State of Charge (%)", row=2, col=1)
    fig_battery.update_yaxes(title_text="Power (kW)", row=3, col=1)
    
    fig_battery.update_layout(height=800, title_text=f"{selected_company.name} - Battery Storage & EV Charging Analysis")
    st.plotly_chart(fig_battery, use_container_width=True)
    
    # Profit History Chart
    if selected_company.profit_history:
        st.subheader("Profit History (Last 24 Hours)")
        
        # Create profit history chart
        hours_back = len(selected_company.profit_history)
        time_labels = [(datetime.now() - timedelta(hours=h)).strftime('%H:%M') 
                      for h in range(hours_back-1, -1, -1)]
        
        fig_profit = go.Figure()
        
        # Add profit line
        fig_profit.add_trace(go.Scatter(
            x=time_labels,
            y=selected_company.profit_history,
            mode='lines+markers',
            name='Hourly Profit',
            line=dict(color='green', width=3),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.1)' if np.mean(selected_company.profit_history) >= 0 else 'rgba(255,0,0,0.1)'
        ))
        
        # Add zero line
        fig_profit.add_hline(y=0, line_dash="dash", line_color="gray", 
                           annotation_text="Break-even")
        
        # Add cumulative profit as secondary trace
        cumulative_values = np.cumsum(selected_company.profit_history)
        fig_profit.add_trace(go.Scatter(
            x=time_labels,
            y=cumulative_values,
            mode='lines',
            name='Cumulative Profit',
            line=dict(color='blue', width=2, dash='dot'),
            yaxis='y2'
        ))
        
        fig_profit.update_layout(
            title=f"{selected_company.name} - Profit Performance",
            xaxis_title="Time",
            yaxis_title="Hourly Profit ($)",
            yaxis2=dict(
                title="Cumulative Profit ($)",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_profit, use_container_width=True)
    
    # Alerts for this company
    company_alerts = [alert for alert in ems.maintenance_alerts 
                     if alert['company'] == selected_company.name]
    
    if company_alerts:
        st.subheader("Active Alerts")
        for alert in company_alerts:
            if alert['type'] == 'CRITICAL':
                st.markdown(f'<div class="alert-critical"><strong>CRITICAL:</strong> {alert["message"]}</div>', 
                           unsafe_allow_html=True)
            elif alert['type'] == 'WARNING':
                st.markdown(f'<div class="alert-warning"><strong>WARNING:</strong> {alert["message"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-maintenance"><strong>MAINTENANCE:</strong> {alert["message"]}</div>', 
                           unsafe_allow_html=True)
    


elif page == "Renewable Market":
    st.header("Renewable Energy Market")
    
    # Market overview
    st.subheader("Market Participants")
    
    market_data = []
    for company in ems.companies:
        total_gen = sum(station.current_generation for station in company.stations)
        total_load_comp = sum(station.current_load for station in company.stations)
        market_data.append({
            'Company': company.name,
            'Generation (kW)': total_gen,
            'Load (kW)': total_load_comp,
            'Balance (kW)': total_gen - total_load_comp,
            'Price ($/kWh)': company.energy_price,
            'Status': 'Seller' if company.total_surplus > 0 else 'Buyer' if company.total_deficit > 0 else 'Balanced'
        })
    
    df_market = pd.DataFrame(market_data)
    st.dataframe(df_market, use_container_width=True)
    
    # Price trends (simulated)
    st.subheader("Price Trends")
    hours = list(range(24))
    price_trends = {
        'Hour': hours,
        'Grid Price': [ems.grid_price + np.sin(h/24 * 2 * np.pi) * 0.02 for h in hours],
        'Company 1': [ems.companies[0].energy_price + np.sin(h/24 * 2 * np.pi) * 0.01 for h in hours],
        'Company 2': [ems.companies[1].energy_price + np.sin((h+4)/24 * 2 * np.pi) * 0.015 for h in hours],
        'Company 3': [ems.companies[2].energy_price + np.sin((h+8)/24 * 2 * np.pi) * 0.012 for h in hours]
    }
    
    df_trends = pd.DataFrame(price_trends)
    fig_trends = px.line(df_trends, x='Hour', y=['Grid Price', 'Company 1', 'Company 2', 'Company 3'],
                        title="Energy Price Trends (24 Hour Forecast)")
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Trading opportunities
    st.subheader("Trading Opportunities")
    
    sellers = [c for c in ems.companies if c.total_surplus > 0]
    buyers = [c for c in ems.companies if c.total_deficit > 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Energy (Sellers)**")
        if sellers:
            seller_data = [(s.name, s.total_surplus, s.energy_price) for s in sellers]
            df_sellers = pd.DataFrame(seller_data, columns=['Company', 'Available (kW)', 'Price ($/kWh)'])
            st.dataframe(df_sellers, use_container_width=True)
        else:
            st.info("No sellers available")
    
    with col2:
        st.write("**Energy Demand (Buyers)**")
        if buyers:
            buyer_data = [(b.name, b.total_deficit, b.energy_price) for b in buyers]
            df_buyers = pd.DataFrame(buyer_data, columns=['Company', 'Needed (kW)', 'Offering ($/kWh)'])
            st.dataframe(df_buyers, use_container_width=True)
        else:
            st.info("No buyers in market")

elif page == "Profit Forecast":
    st.header("24-Hour Profit Forecast")
    
    # Company selector for profit forecast
    company_names = [company.name for company in ems.companies]
    selected_company_name = st.selectbox("Select Company for Profit Forecast", company_names, key="profit_company")
    selected_company = next(c for c in ems.companies if c.name == selected_company_name)
    
    # Get 24-hour forecast data
    profit_forecast, load_forecast, generation_forecast = ems.get_profit_forecast_24h(selected_company)
    
    # Create time labels for next 24 hours
    current_time = datetime.now()
    time_labels = [(current_time + timedelta(hours=h)).strftime('%H:%M') for h in range(24)]
    
    # Key forecast metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_forecast_profit = sum(profit_forecast)
        profit_color = "normal" if total_forecast_profit >= 0 else "inverse"
        st.metric("24h Total Profit", f"${total_forecast_profit:.2f}", delta_color=profit_color)
    
    with col2:
        avg_hourly_profit = np.mean(profit_forecast)
        st.metric("Avg Hourly Profit", f"${avg_hourly_profit:.2f}")
    
    with col3:
        max_profit_hour = np.argmax(profit_forecast)
        st.metric("Best Hour", f"{time_labels[max_profit_hour]}", 
                 f"${profit_forecast[max_profit_hour]:.2f}")
    
    with col4:
        min_profit_hour = np.argmin(profit_forecast)
        st.metric("Worst Hour", f"{time_labels[min_profit_hour]}", 
                 f"${profit_forecast[min_profit_hour]:.2f}")
    
    # Create comprehensive forecast charts
    
    # Chart 1: Load vs Generation with Surplus/Deficit highlighting
    st.subheader("⚡ Energy Balance Forecast")
    
    fig_energy = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Load vs Generation', 'Energy Balance (Surplus/Deficit)'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Add load and generation traces
    fig_energy.add_trace(
        go.Scatter(x=time_labels, y=load_forecast, name='Predicted Load', 
                  line=dict(color='red', width=3), marker=dict(size=6)),
        row=1, col=1
    )
    
    fig_energy.add_trace(
        go.Scatter(x=time_labels, y=generation_forecast, name='Predicted Generation', 
                  line=dict(color='green', width=3), marker=dict(size=6)),
        row=1, col=1
    )
    
    # Calculate and add energy balance
    energy_balance = [gen - load for gen, load in zip(generation_forecast, load_forecast)]
    
    # Create color array for surplus (green) and deficit (red)
    colors = ['green' if balance >= 0 else 'red' for balance in energy_balance]
    
    fig_energy.add_trace(
        go.Bar(x=time_labels, y=energy_balance, name='Energy Balance',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # Add zero line to balance chart
    fig_energy.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig_energy.update_layout(
        height=600,
        title=f"{selected_company.name} - 24-Hour Energy Forecast",
        hovermode='x unified'
    )
    
    fig_energy.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig_energy.update_yaxes(title_text="Balance (kW)", row=2, col=1)
    fig_energy.update_xaxes(title_text="Time", row=2, col=1)
    
    st.plotly_chart(fig_energy, use_container_width=True)
    
    # Chart 2: Profit Forecast
    st.subheader("💰 Profit Forecast")
    
    fig_profit = go.Figure()
    
    # Add profit bars with color coding
    profit_colors = ['green' if p >= 0 else 'red' for p in profit_forecast]
    
    fig_profit.add_trace(go.Bar(
        x=time_labels,
        y=profit_forecast,
        name='Hourly Profit',
        marker_color=profit_colors,
        opacity=0.8,
        text=[f"${p:.2f}" for p in profit_forecast],
        textposition='outside'
    ))
    
    # Add cumulative profit line
    cumulative_profit = np.cumsum(profit_forecast)
    fig_profit.add_trace(go.Scatter(
        x=time_labels,
        y=cumulative_profit,
        mode='lines+markers',
        name='Cumulative Profit',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Add zero line
    fig_profit.add_hline(y=0, line_dash="dash", line_color="gray", 
                        annotation_text="Break-even")
    
    fig_profit.update_layout(
        title=f"{selected_company.name} - 24-Hour Profit Forecast",
        xaxis_title="Time",
        yaxis_title="Hourly Profit ($)",
        yaxis2=dict(
            title="Cumulative Profit ($)",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_profit, use_container_width=True)
    
    # Forecast summary table
    st.subheader("📊 Detailed Forecast Summary")
    
    forecast_df = pd.DataFrame({
        'Hour': time_labels,
        'Load (kW)': [f"{load:.2f}" for load in load_forecast],
        'Generation (kW)': [f"{gen:.2f}" for gen in generation_forecast],
        'Balance (kW)': [f"{balance:.2f}" for balance in energy_balance],
        'Status': ['Surplus' if balance >= 0 else 'Deficit' for balance in energy_balance],
        'Hourly Profit ($)': [f"${profit:.2f}" for profit in profit_forecast],
        'Cumulative Profit ($)': [f"${cum:.2f}" for cum in cumulative_profit]
    })
    
    # Add styling to the dataframe
    def highlight_status(row):
        if row['Status'] == 'Surplus':
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)
    
    styled_df = forecast_df.style.apply(highlight_status, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    surplus_hours = sum(1 for balance in energy_balance if balance >= 0)
    deficit_hours = 24 - surplus_hours
    peak_surplus = max(energy_balance) if max(energy_balance) > 0 else 0
    peak_deficit = abs(min(energy_balance)) if min(energy_balance) < 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚡ Energy Profile:**")
        st.write(f"• Surplus Hours: {surplus_hours}/24 ({surplus_hours/24*100:.1f}%)")
        st.write(f"• Deficit Hours: {deficit_hours}/24 ({deficit_hours/24*100:.1f}%)")
        st.write(f"• Peak Surplus: {peak_surplus:.2f} kW")
        st.write(f"• Peak Deficit: {peak_deficit:.2f} kW")
        st.write(f"• Average Load: {np.mean(load_forecast):.2f} kW")
        st.write(f"• Average Generation: {np.mean(generation_forecast):.2f} kW")
    
    with col2:
        st.markdown("**💰 Financial Profile:**")
        profitable_hours = sum(1 for p in profit_forecast if p >= 0)
        loss_hours = 24 - profitable_hours
        st.write(f"• Profitable Hours: {profitable_hours}/24 ({profitable_hours/24*100:.1f}%)")
        st.write(f"• Loss Hours: {loss_hours}/24 ({loss_hours/24*100:.1f}%)")
        st.write(f"• Best Hour Profit: ${max(profit_forecast):.2f}")
        st.write(f"• Worst Hour Loss: ${min(profit_forecast):.2f}")
        breakeven_index = next((i for i, p in enumerate(cumulative_profit) if p >= 0), None)
        breakeven_hour = time_labels[breakeven_index] if breakeven_index is not None else "Never"
        st.write(f"• Break-even at Hour: {breakeven_hour}")
        st.write(f"• Final Position: ${cumulative_profit[-1]:.2f}")

elif page == "System Settings":
    st.header("System Settings")
    
    # Algorithm settings
    st.subheader("Algorithm Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**RL Algorithm Settings**")
        new_learning_rate = st.slider("Learning Rate", 0.001, 0.1, ems.learning_rate, 0.001)
        new_epsilon = st.slider("Exploration Rate (ε)", 0.0, 1.0, ems.epsilon, 0.01)
        
        if st.button("Update RL Settings"):
            ems.learning_rate = new_learning_rate
            ems.epsilon = new_epsilon
            st.success("RL settings updated!")
    
    with col2:
        st.write("**Safety Thresholds**")
        new_critical_soc = st.slider("Battery Critical SOC (%)", 5.0, 20.0, ems.battery_critical_soc, 1.0)
        new_low_soc = st.slider("Battery Low SOC (%)", 15.0, 30.0, ems.battery_low_soc, 1.0)
        new_inverter_threshold = st.slider("Inverter Efficiency Threshold (%)", 70.0, 95.0, ems.inverter_efficiency_threshold, 1.0)
        new_solar_threshold = st.slider("Solar Panel Efficiency Threshold (%)", 60.0, 90.0, ems.solar_efficiency_threshold, 1.0)
        
        if st.button("Update Safety Thresholds"):
            ems.battery_critical_soc = new_critical_soc
            ems.battery_low_soc = new_low_soc
            ems.inverter_efficiency_threshold = new_inverter_threshold
            ems.solar_efficiency_threshold = new_solar_threshold
            st.success("Safety thresholds updated!")
    
    # Manual control section
    if ems.operation_mode == OperationMode.MANUAL:
        st.subheader("Manual Control")
        
        selected_company_manual = st.selectbox("Select Company for Manual Control", 
                                             [c.name for c in ems.companies])
        company_manual = next(c for c in ems.companies if c.name == selected_company_manual)
        
        if company_manual.total_surplus > 0:
            st.write(f"Company has surplus: {company_manual.total_surplus:.1f} kW")
            
            # Manual trading options
            trade_amount = st.number_input("Amount to trade (kW)", 
                                         min_value=0.0, 
                                         max_value=company_manual.total_surplus, 
                                         value=company_manual.total_surplus/2)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Sell to Grid"):
                    ems._log_trade(company_manual.name, "Grid", trade_amount, ems.grid_price)
                    st.success(f"Sold {trade_amount:.1f} kW to Grid at ${ems.grid_price:.4f}/kWh")
            
            with col2:
                # Find potential buyers
                buyers = [c for c in ems.companies if c.id != company_manual.id and c.total_deficit > 0]
                if buyers:
                    buyer_names = [b.name for b in buyers]
                    selected_buyer = st.selectbox("Select Buyer", buyer_names)
                    if st.button("Sell to Company"):
                        buyer = next(b for b in buyers if b.name == selected_buyer)
                        ems._log_trade(company_manual.name, buyer.name, trade_amount, buyer.energy_price)
                        st.success(f"Sold {trade_amount:.1f} kW to {buyer.name} at ${buyer.energy_price:.4f}/kWh")
        
        elif company_manual.total_deficit > 0:
            st.write(f"Company has deficit: {company_manual.total_deficit:.1f} kW")
            
            # Manual purchase options
            purchase_amount = st.number_input("Amount to purchase (kW)", 
                                            min_value=0.0, 
                                            max_value=company_manual.total_deficit, 
                                            value=company_manual.total_deficit/2)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Buy from Grid"):
                    ems._log_trade("Grid", company_manual.name, purchase_amount, ems.grid_price)
                    st.success(f"Bought {purchase_amount:.1f} kW from Grid at ${ems.grid_price:.4f}/kWh")
            
            with col2:
                # Find potential sellers
                sellers = [c for c in ems.companies if c.id != company_manual.id and c.total_surplus > 0]
                if sellers:
                    seller_names = [s.name for s in sellers]
                    selected_seller = st.selectbox("Select Seller", seller_names)
                    if st.button("Buy from Company"):
                        seller = next(s for s in sellers if s.name == selected_seller)
                        ems._log_trade(seller.name, company_manual.name, purchase_amount, seller.energy_price)
                        st.success(f"Bought {purchase_amount:.1f} kW from {seller.name} at ${seller.energy_price:.4f}/kWh")
        else:
            st.info("Company is balanced - no trading needed")
    
    # System status
    # Real-time ingest UI
    st.subheader("Real-time Ingest")
    st.markdown("Send a JSON payload representing an IoT device update to the ingestion pipeline.")
    # Default payload helper
    default_payload = json.dumps({"device_id": "BAT_1_1_1", "soc": 55.0, "temperature": 30.0}, indent=2)

    with st.form("ingest_form"):
        # Determine available delivery methods
        methods = []
        if backend is not None and getattr(backend, 'manager', None) is not None:
            methods.append("Local Manager")
        # HTTP API is always an option (may fail if service not running)
        methods.append("HTTP API")

        delivery = st.radio("Delivery method", options=methods)
        device_input = st.text_input("Device ID (for convenience)", value="BAT_1_1_1")
        payload_text = st.text_area("Payload (JSON)", value=default_payload, height=180)
        submitted = st.form_submit_button("Send Payload")

    if submitted:
        try:
            payload = json.loads(payload_text)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            payload = None

        if payload is not None:
            # Ensure device id from quick field is applied if provided
            if device_input and ("device_id" not in payload and "deviceId" not in payload and "id" not in payload):
                payload["device_id"] = device_input

            if delivery == "Local Manager":
                mgr = getattr(backend, 'manager', None)
                if mgr is None:
                    st.error("Local ingest manager not available in this process.")
                else:
                    try:
                        ok = mgr.push_nowait(payload)
                        if ok:
                            st.success("Payload accepted by local manager")
                        else:
                            st.warning("Local manager queue full; payload dropped")
                    except Exception as e:
                        st.error(f"Local manager push failed: {e}")
            else:
                if requests is None:
                    st.error("`requests` library not available in this environment; cannot send HTTP request.")
                else:
                    try:
                        url = st.text_input("Ingest HTTP URL", value="http://localhost:8000/ingest")
                        r = requests.post(url, json=payload, timeout=5)
                        if r.status_code in (200, 202):
                            st.success(f"HTTP ingest accepted: {r.status_code}")
                        else:
                            st.error(f"HTTP ingest failed: {r.status_code} {r.text}")
                    except Exception as e:
                        st.error(f"HTTP request failed: {e}")

    # Data entry section for repository CSVs
    st.subheader("Manual Data Entry")
    st.markdown("You can append rows to the local data files used by the models. Use with care — this writes directly to files in the `data/` folder.")
    base_path = os.path.dirname(__file__)
    data_folder = os.path.join(base_path, "data")

    # Hourly EV usage
    with st.expander("Add Hourly EV Usage (hourlyEVusage_cleaned.csv)"):
        ev_date = st.date_input("Date", value=datetime.now().date(), key="ev_date")
        ev_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=0, key="ev_hour")
        ev_energy = st.number_input("Energy (kWh)", min_value=0.0, value=0.0, format="%.6f", key="ev_energy")
        if st.button("Append EV Usage Row"):
            try:
                ev_path = os.path.join(data_folder, "hourlyEVusage_cleaned.csv")
                ev_datetime = f"{ev_date.strftime('%Y-%m-%d')} {int(ev_hour):02d}:00:00"
                row = {
                    "Date": ev_date.strftime('%Y-%m-%d'),
                    "Hour": int(ev_hour),
                    "Energy_kWh": float(ev_energy),
                    "Datetime": ev_datetime
                }
                import pandas as _pd
                df_row = _pd.DataFrame([row])
                write_header = not os.path.exists(ev_path)
                df_row.to_csv(ev_path, mode='a', header=write_header, index=False)
                # Audit the manual append
                try:
                    audit_path = os.path.join(data_folder, "manual_appends_audit.csv")
                    import pandas as _pd
                    audit_row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "user": "dashboard",
                        "target_file": "hourlyEVusage_cleaned.csv",
                        "appended_row": str(row),
                        "result": "success"
                    }
                    _pd.DataFrame([audit_row]).to_csv(audit_path, mode='a', header=not os.path.exists(audit_path), index=False)
                except Exception:
                    pass

                st.success(f"Appended EV usage row to {ev_path}")
                st.json(row)
            except Exception as e:
                st.error(f"Failed to append EV usage row: {e}")

    # Satellite data
    with st.expander("Add Satellite Hourly Data (satelite_hourly_data.csv)"):
        sat_date = st.date_input("Date", value=datetime.now().date(), key="sat_date")
        sat_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=6, key="sat_hour")
        solar_rad = st.number_input("Solar radiation", min_value=0.0, value=0.0, format="%.6f", key="solar_rad")
        module_temp = st.number_input("Module temp", value=0.0, format="%.3f", key="module_temp")
        wind_dir = st.number_input("Wind direction", value=0.0, format="%.3f", key="wind_dir")
        wind_speed = st.number_input("Wind speed", value=0.0, format="%.3f", key="wind_speed")
        power_val = st.number_input("Power", value=0.0, format="%.6f", key="power_val")
        if st.button("Append Satellite Row"):
            try:
                sat_path = os.path.join(data_folder, "satelite_hourly_data.csv")
                dy = int(sat_date.day)
                mo = int(sat_date.month)
                row = {
                    "DY": dy,
                    "MO": mo,
                    "HR": int(sat_hour),
                    "solar radiation": float(solar_rad),
                    "module_temp": float(module_temp),
                    "wind direction": float(wind_dir),
                    "wind speed": float(wind_speed),
                    "power": float(power_val)
                }
                import pandas as _pd
                df_row = _pd.DataFrame([row])
                write_header = not os.path.exists(sat_path)
                df_row.to_csv(sat_path, mode='a', header=write_header, index=False)
                # Audit the manual append
                try:
                    audit_path = os.path.join(data_folder, "manual_appends_audit.csv")
                    import pandas as _pd
                    audit_row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "user": "dashboard",
                        "target_file": "satelite_hourly_data.csv",
                        "appended_row": str(row),
                        "result": "success"
                    }
                    _pd.DataFrame([audit_row]).to_csv(audit_path, mode='a', header=not os.path.exists(audit_path), index=False)
                except Exception:
                    pass

                st.success(f"Appended satellite data row to {sat_path}")
                st.json(row)
            except Exception as e:
                st.error(f"Failed to append satellite data row: {e}")
    st.subheader("System Status")
    st.json({
        "Total Companies": len(ems.companies),
        "Total Stations": sum(len(c.stations) for c in ems.companies),
        "Total Batteries": sum(len(s.batteries) for c in ems.companies for s in c.stations),
        "Total Inverters": sum(len(s.inverters) for c in ems.companies for s in c.stations),
        "Total Solar Panels": sum(len(s.solar_panels) for c in ems.companies for s in c.stations),
        "Active Alerts": len(ems.maintenance_alerts),
        "Trading Log Entries": len(ems.trading_log),
        "RL Q-Table Size": len(ems.q_table)
    })

# Footer
st.markdown("---")
st.markdown("*Last Updated: " + st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S") + "*")

# Real-time updates notification
if auto_refresh:
    time.sleep(1)  # Small delay to prevent too frequent updates
