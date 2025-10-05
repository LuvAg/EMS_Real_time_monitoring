# Smart Energy Management Dashboard

A comprehensive energy management system for a network of companies with renewable energy trading capabilities, built with Streamlit and featuring both Rule-based and Reinforcement Learning algorithms.

## 🌟 Features

### Core Functionality
- **Multi-Company Network Management**: Manage multiple companies, each with their own charging stations
- **Real-time Energy Trading**: Automated energy trading between companies and grid
- **Battery Management**: Intelligent battery charging/discharging based on SOC and SOH
- **Predictive Analytics**: Integration with existing ML models for load and generation forecasting
- **Maintenance Monitoring**: Automated alerts for battery, inverter, and solar panel maintenance

### Algorithm Options
- **Hybrid RL Algorithm**: Advanced reinforcement learning with experience replay
- **Rule-based Algorithm**: Traditional if-else logic for reliable operation
- **Manual Override**: Complete manual control when needed

### Dashboard Pages
1. **Market Overview**: System-wide energy balance and trading activity
2. **Your Company**: Detailed view of selected company's stations and components
3. **Renewable Market**: Energy trading opportunities and price trends
4. **Grid Cost**: Grid price analysis and cost comparisons
5. **System Settings**: Algorithm configuration and manual controls

## 🏗️ System Architecture

### Components Hierarchy
```
Companies
├── Charging Stations
│   ├── Batteries (SOC, SOH, Capacity)
│   ├── Inverters (Efficiency, Temperature)
│   └── Solar Panels (Efficiency, Generation)
└── Energy Trading (Prices, Surplus/Deficit)
```

### Algorithm Comparison: RL vs Rule-Based

#### **Reinforcement Learning Approach**
✅ **Best for**: Dynamic optimization, learning from market patterns
- **Advantages**:
  - Adapts to changing market conditions
  - Optimizes long-term rewards
  - Learns from trading patterns
  - Better performance over time

- **Disadvantages**:
  - Requires training period
  - Less predictable initially
  - More complex to debug

#### **Rule-Based Approach**  
✅ **Best for**: Predictable, safety-critical operations
- **Advantages**:
  - Immediately operational
  - Transparent decision logic
  - Reliable for emergency situations
  - Easy to modify and debug

- **Disadvantages**:
  - Fixed decision patterns
  - No learning capability
  - May miss optimal opportunities

### **Recommended Hybrid Approach** 🎯
The system uses a **hybrid approach** combining both:
- **RL for energy trading optimization** (70% weight)
- **Rule-based for safety operations** (30% weight)
- **Rule-based override for emergencies** (100% when critical)

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Quick Start
1. **Clone/Download** the project to your desired directory
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the dashboard**:
   ```bash
   streamlit run energy_dashboard.py
   ```
   Or use the provided batch file:
   ```bash
   start_dashboard.bat
   ```
4. **Open your browser** to `http://localhost:8501`

### Configuration
- All system parameters can be adjusted in the **System Settings** page
- Algorithm weights and thresholds are configurable
- Manual override available for all operations

## 📊 Key Metrics & Alerts

### Battery Management
- **Critical SOC**: < 10% (Emergency charging triggered)
- **Low SOC**: < 20% (Warning alert)
- **SOH Threshold**: < 70% (Maintenance required)

### Component Monitoring
- **Inverter Efficiency**: < 85% (Maintenance alert)
- **Solar Panel Efficiency**: < 80% (Maintenance alert)
- **Temperature Monitoring**: All components tracked

### Trading Logic
- **Surplus Energy**: Sell to highest bidder (companies vs grid)
- **Deficit Energy**: Buy from cheapest source
- **Emergency Charging**: Immediate charging for critical batteries
- **Price Competition**: Dynamic pricing based on supply/demand

## 🔧 System Operations

### Automatic Mode (Recommended)
- **RL Algorithm**: Optimizes trading decisions based on learned patterns
- **Rule-based Safety**: Handles emergencies and critical situations
- **Hourly Updates**: System updates every hour with new predictions
- **Real-time Monitoring**: Continuous monitoring of all components

### Manual Mode
- **Complete Control**: Override all automatic decisions
- **Individual Trading**: Manual energy buying/selling
- **Component Control**: Direct battery charging control
- **Emergency Response**: Manual trigger for maintenance actions

## 📈 Energy Trading Algorithm

### RL Algorithm Details
```python
State Space:
- Company surplus/deficit
- Grid price
- Competitor prices
- Time of day
- Battery states

Action Space:
- Hold (do nothing)
- Sell to grid
- Buy from grid
- Sell to company X
- Buy from company X
- Emergency charge

Reward Function:
- Revenue from sales (+)
- Cost of purchases (-)
- Battery health bonus (+)
- Maintenance penalties (-)
- Emergency response rewards (+)
```

### Rule-Based Logic
```python
Priority Order:
1. Emergency battery charging (SOC < 10%)
2. Sell surplus to highest bidder
3. Buy deficit from cheapest source
4. Maintain optimal battery SOC levels
5. Monitor component health
```

## 🛠️ Integration with Existing Models

The system integrates with your existing ML models:
- **Power Prediction**: Uses your PyTorch LSTM models (`powerpredict.pkl`)
- **Load Forecasting**: CatBoost models for EV load prediction
- **Data Pipeline**: Connects to your existing data sources

### Model Files Used
- `model/powerpredict.pkl` - Power generation forecasting
- `model/catboost_ev_forecast.cbm` - EV load forecasting
- `model/scaler_*.pkl` - Data preprocessing scalers

## 📁 File Structure
```
VidyutAI/
├── energy_dashboard.py          # Main Streamlit dashboard
├── energy_management_system.py  # Core system logic
├── advanced_rl_agent.py        # RL implementation
├── requirements.txt             # Dependencies
├── start_dashboard.bat         # Windows startup script
├── model/                      # Your existing ML models
│   ├── powerpredict.pkl
│   ├── catboost_ev_forecast.cbm
│   └── scaler_*.pkl
└── data/                       # Data files
    ├── hourlyEVusage_cleaned.csv
    └── satelite_hourly_data.csv
```

## 🔍 Monitoring & Alerts

### Alert Types
- 🔴 **CRITICAL**: Battery SOC < 10% (Immediate action required)
- 🟡 **WARNING**: Battery SOC < 20% (Monitor closely)
- 🟢 **MAINTENANCE**: Component efficiency below threshold

### Dashboard Features
- **Real-time Updates**: Auto-refresh every 30 seconds
- **Historical Data**: View past trading activities
- **Performance Metrics**: Track RL learning progress
- **Component Status**: All batteries, inverters, solar panels

## 🎛️ Advanced Features

### RL Model Training
- **Experience Replay**: Learns from historical decisions
- **Epsilon-Greedy**: Balances exploration vs exploitation
- **Q-Learning**: Updates policy based on rewards
- **Model Persistence**: Saves/loads trained models

### Safety Features
- **Emergency Override**: Rule-based control for critical situations
- **Threshold Monitoring**: Configurable safety limits
- **Alert System**: Real-time notifications for all issues
- **Manual Control**: Complete operator override capability

## 🔮 Future Enhancements

### Planned Features
- **Weather Integration**: Solar generation forecasting
- **Market Prediction**: Price forecasting algorithms
- **Multi-objective Optimization**: Balance profit vs sustainability
- **Mobile Dashboard**: Responsive design for mobile devices
- **API Integration**: Real-time grid price feeds
- **Advanced RL**: Deep Q-Networks (DQN) implementation

### Scalability
- **Cloud Deployment**: Easy deployment to cloud platforms
- **Database Integration**: PostgreSQL/MongoDB support
- **Microservices**: Split into independent services
- **Load Balancing**: Handle multiple company networks

## 📞 Support & Documentation

### Key Commands
```bash
# Start dashboard
streamlit run energy_dashboard.py

# Install dependencies
pip install -r requirements.txt

# Update RL model
python -c "from advanced_rl_agent import HybridEnergyAgent; agent = HybridEnergyAgent(); agent.save_model()"
```

### Troubleshooting
1. **Import Errors**: Check if all dependencies are installed
2. **Model Loading**: Ensure model files exist in correct paths
3. **Data Issues**: Verify CSV files are properly formatted
4. **RL Training**: Allow time for RL algorithm to learn patterns

### Performance Tips
- **Batch Size**: Adjust RL batch size for your system memory
- **Update Frequency**: Balance real-time updates vs system load
- **Model Complexity**: Start simple, add complexity gradually
- **Data Quality**: Ensure high-quality training data for better RL performance

---

## 📊 System Dashboard Preview

The dashboard provides:
- **📈 Real-time Metrics**: Energy generation, consumption, trading
- **🏢 Company Overview**: Multi-company management interface  
- **🔋 Component Details**: Battery, inverter, solar panel status
- **💰 Trading Activity**: Live trading logs and opportunities
- **⚙️ Configuration**: Algorithm settings and manual controls
- **🚨 Alert Management**: Real-time maintenance notifications

**Access the dashboard at**: `http://localhost:8501` after running the startup command.

This system provides a complete solution for managing renewable energy networks with intelligent trading capabilities, combining the reliability of rule-based systems with the optimization power of reinforcement learning.