import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')
# Optional import for ingest manager integration
try:
    from backend.ingest_manager import IngestManager
except Exception:
    IngestManager = None

# Import advanced RL agent
try:
    from advanced_rl_agent import HybridEnergyAgent
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Advanced RL agent not available, using basic implementation")

# Import energy prediction service
try:
    from energy_prediction_service import EnergyPredictionService
    PREDICTION_SERVICE_AVAILABLE = True
except ImportError:
    PREDICTION_SERVICE_AVAILABLE = False
    print("Energy prediction service not available, using random predictions")

# Data Classes for System Components
@dataclass
class Battery:
    id: str
    soc: float  # State of Charge (0-100%)
    soh: float  # State of Health (0-100%)
    capacity: float  # kWh
    current_power: float  # Current charging/discharging power
    temperature: float
    voltage: float
    maintenance_required: bool = False

@dataclass
class Inverter:
    id: str
    efficiency: float  # 0-100%
    temperature: float
    power_rating: float  # kW
    current_load: float
    maintenance_required: bool = False

@dataclass
class SolarPanel:
    id: str
    efficiency: float  # 0-100%
    power_rating: float  # kW
    current_generation: float
    temperature: float
    maintenance_required: bool = False

@dataclass
class ChargingStation:
    id: str
    name: str
    location: str
    batteries: List[Battery]
    inverters: List[Inverter]
    solar_panels: List[SolarPanel]
    predicted_load: float  # Next hour prediction
    predicted_generation: float  # Next hour solar prediction
    current_load: float
    current_generation: float

@dataclass
class Company:
    id: str
    name: str
    stations: List[ChargingStation]
    energy_price: float  # Current selling price
    total_surplus: float
    total_deficit: float
    # Profit tracking
    current_hour_profit: float = 0.0
    cumulative_profit: float = 0.0
    expected_next_hour_profit: float = 0.0
    profit_history: List[float] = None
    
    def __post_init__(self):
        if self.profit_history is None:
            self.profit_history = []

class OperationMode(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"

class EnergyManagementSystem:
    def __init__(self, ingest_manager=None):
        self.grid_price = self._get_grid_price()
        self.operation_mode = OperationMode.AUTOMATIC
        self.trading_log = []
        self.maintenance_alerts = []
        
        # RL Environment Parameters
        self.rl_enabled = True
        self.learning_rate = 0.01
        self.epsilon = 0.1  # Exploration rate
        self.q_table = {}  # Simplified Q-table for demonstration
        
        # Initialize advanced RL agent if available
        if RL_AVAILABLE:
            self.rl_agent = HybridEnergyAgent()
        else:
            self.rl_agent = None
        
        # Initialize energy prediction service first
        if PREDICTION_SERVICE_AVAILABLE:
            self.prediction_service = EnergyPredictionService()
            print("Energy prediction service initialized")
        else:
            self.prediction_service = None
            print("Energy prediction service not available")
        
        # Initialize companies after prediction service is ready
        self.companies = self._initialize_companies()

        # Optional ingest manager integration (synchronous callback)
        self.ingest_manager = None
        if ingest_manager is not None and IngestManager is not None:
            try:
                self.register_ingest_manager(ingest_manager)
            except Exception:
                # ignore integration failure and continue
                self.ingest_manager = None
        
        # Thresholds
        self.battery_critical_soc = 10.0  # %
        self.battery_low_soc = 20.0  # %
        self.inverter_efficiency_threshold = 85.0  # %
        self.solar_efficiency_threshold = 80.0  # %
        
    def _initialize_companies(self) -> List[Company]:
        """Initialize sample companies with charging stations"""
        companies = []
        
        for company_id in range(1, 4):  # 3 companies
            stations = []
            for station_id in range(1, 4):  # 3 stations per company
                # Generate batteries
                batteries = []
                for bat_id in range(1, 5):  # 4 batteries per station
                    battery = Battery(
                        id=f"BAT_{company_id}_{station_id}_{bat_id}",
                        soc=random.uniform(20, 90),
                        soh=random.uniform(80, 100),
                        capacity=random.uniform(10, 25),  # Reduced to achieve ~200kWh total per company
                        current_power=random.uniform(-10, 10),
                        temperature=random.uniform(25, 35),
                        voltage=random.uniform(400, 450)
                    )
                    batteries.append(battery)
                
                # Generate inverters
                inverters = []
                for inv_id in range(1, 3):  # 2 inverters per station
                    inverter = Inverter(
                        id=f"INV_{company_id}_{station_id}_{inv_id}",
                        efficiency=random.uniform(80, 95),
                        temperature=random.uniform(30, 50),
                        power_rating=random.uniform(50, 100),
                        current_load=random.uniform(20, 80)
                    )
                    inverters.append(inverter)
                
                # Generate solar panels
                solar_panels = []
                for panel_id in range(1, 6):  # 5 panels per station
                    panel = SolarPanel(
                        id=f"SOLAR_{company_id}_{station_id}_{panel_id}",
                        efficiency=random.uniform(75, 90),
                        power_rating=random.uniform(5, 15),
                        current_generation=random.uniform(2, 12),
                        temperature=random.uniform(35, 55)
                    )
                    solar_panels.append(panel)
                
                # Get initial predictions for this station
                if self.prediction_service:
                    try:
                        current_load, current_gen = self.prediction_service.get_current_predictions(f"STATION_{company_id}_{station_id}")
                        predicted_load, predicted_gen = self.prediction_service.get_next_hour_predictions(f"STATION_{company_id}_{station_id}")
                    except Exception as e:
                        print(f"⚠️ Using fallback values for station initialization: {e}")
                        current_load, current_gen = 75.0, 45.0
                        predicted_load, predicted_gen = 80.0, 50.0
                else:
                    # Fallback values
                    current_load, current_gen = random.uniform(40, 140), random.uniform(25, 95)
                    predicted_load, predicted_gen = random.uniform(50, 150), random.uniform(30, 100)
                
                station = ChargingStation(
                    id=f"STATION_{company_id}_{station_id}",
                    name=f"Station {station_id}",
                    location=f"Location {company_id}-{station_id}",
                    batteries=batteries,
                    inverters=inverters,
                    solar_panels=solar_panels,
                    predicted_load=predicted_load,
                    predicted_generation=predicted_gen,
                    current_load=current_load,
                    current_generation=current_gen
                )
                stations.append(station)
            
            company = Company(
                id=f"COMPANY_{company_id}",
                name=f"Company {company_id}",
                stations=stations,
                energy_price=random.uniform(0.08, 0.15),  # $/kWh
                total_surplus=0.0,
                total_deficit=0.0
            )
            companies.append(company)
        
        return companies
    
    def _get_grid_price(self) -> float:
        """Get current grid price (simulated API call)"""
        # In real implementation, this would call an actual API
        # For demo, we'll simulate with random price
        base_price = 0.12  # $/kWh
        variation = random.uniform(-0.02, 0.03)
        return max(0.05, base_price + variation)

    def register_ingest_manager(self, manager):
        """Register an IngestManager instance so incoming IoT messages
        are forwarded into the EnergyManagementSystem via a synchronous callback.
        This keeps the ingestion API simple and avoids managing background
        asyncio tasks inside this class.
        """
        self.ingest_manager = manager
        # register a synchronous callback - IngestManager will call it for each message
        try:
            manager.register_callback(self._sync_ingest_handler)
        except Exception:
            # For safety, ignore registration errors
            pass

    def _sync_ingest_handler(self, payload: Dict[str, Any]):
        """Handle a single incoming IoT payload and update matching components.
        This function is intended to be called synchronously from the IngestManager
        callbacks and must be fast / non-blocking.
        Expected payload keys (examples):
            {"deviceId": "BAT_1_2_3", "soc": 45.3, "temperature": 30.1}
        """
        try:
            device = payload.get("deviceId") or payload.get("device_id") or payload.get("id")
            if not device:
                return

            # Simple heuristic: match device id to battery/inverter/panel/station ids
            for company in self.companies:
                for station in company.stations:
                    # Batteries
                    for battery in station.batteries:
                        if battery.id == device:
                            if "soc" in payload:
                                try:
                                    battery.soc = float(payload["soc"])
                                except Exception:
                                    pass
                            if "soh" in payload:
                                try:
                                    battery.soh = float(payload["soh"])
                                except Exception:
                                    pass
                            if "current_power" in payload:
                                try:
                                    battery.current_power = float(payload["current_power"])
                                except Exception:
                                    pass
                            if "temperature" in payload:
                                try:
                                    battery.temperature = float(payload["temperature"])
                                except Exception:
                                    pass
                            if "voltage" in payload:
                                try:
                                    battery.voltage = float(payload["voltage"])
                                except Exception:
                                    pass
                            return

                    # Inverters
                    for inverter in station.inverters:
                        if inverter.id == device:
                            if "efficiency" in payload:
                                try:
                                    inverter.efficiency = float(payload["efficiency"])
                                except Exception:
                                    pass
                            if "current_load" in payload:
                                try:
                                    inverter.current_load = float(payload["current_load"])
                                except Exception:
                                    pass
                            if "temperature" in payload:
                                try:
                                    inverter.temperature = float(payload["temperature"])
                                except Exception:
                                    pass
                            return

                    # Solar panels
                    for panel in station.solar_panels:
                        if panel.id == device:
                            if "current_generation" in payload:
                                try:
                                    panel.current_generation = float(payload["current_generation"])
                                except Exception:
                                    pass
                            if "efficiency" in payload:
                                try:
                                    panel.efficiency = float(payload["efficiency"])
                                except Exception:
                                    pass
                            return

                    # Station-level updates
                    if station.id == device:
                        if "current_load" in payload:
                            try:
                                station.current_load = float(payload["current_load"])
                            except Exception:
                                pass
                        if "current_generation" in payload:
                            try:
                                station.current_generation = float(payload["current_generation"])
                            except Exception:
                                pass
                        return
        except Exception:
            # Everything must be resilient in the callback
            return
    
    def update_system_state(self):
        """Update system state - called every hour"""
        self.grid_price = self._get_grid_price()
        
        for company in self.companies:
            self._update_company_state(company)
            self._calculate_energy_balance(company)
        
        if self.operation_mode == OperationMode.AUTOMATIC:
            if self.rl_enabled:
                self._rl_energy_management()
            else:
                self._rule_based_energy_management()
        
        self._check_maintenance_alerts()
    
    def _update_company_state(self, company: Company):
        """Update individual company state"""
        for station in company.stations:
            # Update predictions using ML models
            if self.prediction_service:
                try:
                    current_load, current_gen = self.prediction_service.get_current_predictions(station.id)
                    predicted_load, predicted_gen = self.prediction_service.get_next_hour_predictions(station.id)
                    
                    station.current_load = current_load
                    station.current_generation = current_gen
                    station.predicted_load = predicted_load
                    station.predicted_generation = predicted_gen
                    
                except Exception as e:
                    print(f"⚠️ Prediction service error for {station.id}: {e}")
                    # Fallback to gradual changes from current values
                    station.current_load += random.uniform(-5, 5)
                    station.current_generation += random.uniform(-3, 3)
                    station.predicted_load = station.current_load + random.uniform(-10, 10)
                    station.predicted_generation = station.current_generation + random.uniform(-5, 5)
            else:
                # Fallback - gradual evolution from current state
                station.current_load += random.uniform(-5, 5)
                station.current_generation += random.uniform(-3, 3)
                station.predicted_load = station.current_load + random.uniform(-10, 10)
                station.predicted_generation = station.current_generation + random.uniform(-5, 5)
            
            # Ensure positive values
            station.current_load = max(0, station.current_load)
            station.current_generation = max(0, station.current_generation)
            station.predicted_load = max(0, station.predicted_load)
            station.predicted_generation = max(0, station.predicted_generation)
            
            # Update solar panel generation to match station total (scaled up from individual panels)
            if station.solar_panels:
                total_panel_generation = station.current_generation
                avg_per_panel = total_panel_generation / len(station.solar_panels)
                for panel in station.solar_panels:
                    # Distribute generation across panels with some variation
                    panel.current_generation = avg_per_panel * random.uniform(0.8, 1.2)
                    panel.efficiency += random.uniform(-1, 0.5)
                    panel.efficiency = max(60, min(95, panel.efficiency))
                    panel.temperature = random.uniform(35, 55)
            
            # Update battery temperature and voltage (physical parameters)
            for battery in station.batteries:
                battery.temperature = random.uniform(25, 35)
                battery.voltage = random.uniform(400, 450)
            
            # Update inverter states
            for inverter in station.inverters:
                inverter.efficiency += random.uniform(-2, 1)
                inverter.efficiency = max(70, min(100, inverter.efficiency))
                inverter.current_load = min(inverter.power_rating, station.current_load / len(station.inverters))
                inverter.temperature = random.uniform(30, 50)
    
    def _calculate_energy_balance(self, company: Company):
        """Calculate energy surplus/deficit for company"""
        # Use station-level generation (already scaled properly from predictions)
        total_generation = sum(station.current_generation for station in company.stations)
        total_load = sum(station.current_load for station in company.stations)
        
        # Add battery contribution to available energy
        battery_contribution = sum(
            sum(max(0, -battery.current_power) for battery in station.batteries)  # Discharging batteries
            for station in company.stations
        )
        
        # Subtract battery charging from available energy
        battery_consumption = sum(
            sum(max(0, battery.current_power) for battery in station.batteries)  # Charging batteries
            for station in company.stations
        )
        
        available_energy = total_generation + battery_contribution
        total_consumption = total_load + battery_consumption
        
        balance = available_energy - total_consumption
        
        if balance > 0:
            company.total_surplus = balance
            company.total_deficit = 0
            # Set competitive price when surplus (5% below grid price to attract buyers)
            company.energy_price = self.grid_price * 0.95
        else:
            company.total_deficit = abs(balance)
            company.total_surplus = 0
            # Set high price when deficit (20% above grid price to discourage selling)
            company.energy_price = self.grid_price * 1.2
        
        # Calculate profits after energy balance
        self._calculate_current_profit(company)
        self._calculate_expected_profit(company)
    
    def _calculate_current_profit(self, company: Company):
        """Calculate current hour profit/loss for company"""
        # Reset current hour profit
        company.current_hour_profit = 0.0
        
        # Calculate profit from battery discharge (avoided grid purchases)
        total_battery_discharge = sum(
            sum(max(0, -battery.current_power) for battery in station.batteries)
            for station in company.stations
        )
        if total_battery_discharge > 0:
            # Battery discharge saves grid purchase costs at grid price
            company.current_hour_profit += total_battery_discharge * self.grid_price
        
        # Calculate profit from energy sales (surplus sold)
        if company.total_surplus > 0:
            # Find best selling price (grid vs other companies)
            best_price = self.grid_price
            for other_company in self.companies:
                if other_company.id != company.id and other_company.total_deficit > 0:
                    if other_company.energy_price > best_price:
                        best_price = other_company.energy_price
            
            company.current_hour_profit += company.total_surplus * best_price
        
        # Calculate cost from energy purchases (deficit bought)
        if company.total_deficit > 0:
            # Find cheapest buying price (grid vs other companies)
            cheapest_price = self.grid_price
            for other_company in self.companies:
                if other_company.id != company.id and other_company.total_surplus > 0:
                    if other_company.energy_price < cheapest_price:
                        cheapest_price = other_company.energy_price
            
            company.current_hour_profit -= company.total_deficit * cheapest_price
        
        # Add operational costs (maintenance, battery degradation, etc.)
        operational_cost = self._calculate_operational_costs(company)
        company.current_hour_profit -= operational_cost
        
        # Update cumulative profit
        company.cumulative_profit += company.current_hour_profit
        
        # Add to profit history
        company.profit_history.append(company.current_hour_profit)
        
        # Keep only last 24 hours of history
        if len(company.profit_history) > 24:
            company.profit_history = company.profit_history[-24:]
    
    def _calculate_expected_profit(self, company: Company):
        """Calculate expected profit for next hour based on predictions"""
        company.expected_next_hour_profit = 0.0
        
        # Get next hour predictions for all stations
        total_predicted_load = sum(station.predicted_load for station in company.stations)
        total_predicted_generation = sum(station.predicted_generation for station in company.stations)
        
        # Estimate potential battery discharge for next hour
        total_available_battery_capacity = sum(
            sum(battery.capacity * (battery.soc / 100.0) * 0.8  # Conservative 80% usable capacity
                for battery in station.batteries if battery.soc > 20)  # Only if above critical level
            for station in company.stations
        )
        
        predicted_balance = total_predicted_generation - total_predicted_load
        
        # If there's a deficit, calculate how much battery can cover
        if predicted_balance < 0:
            deficit = abs(predicted_balance)
            battery_coverage = min(deficit, total_available_battery_capacity)
            # Battery discharge saves grid purchase costs
            company.expected_next_hour_profit += battery_coverage * self.grid_price
            # Remaining deficit after battery usage
            remaining_deficit = deficit - battery_coverage
            if remaining_deficit > 0:
                estimated_cost = self.grid_price * 1.05  # Premium for purchasing
                company.expected_next_hour_profit -= remaining_deficit * estimated_cost
        
        if predicted_balance > 0:  # Expected surplus
            # Estimate selling price (assume similar to current market conditions)
            estimated_price = self.grid_price * 0.95  # Conservative estimate
            company.expected_next_hour_profit += predicted_balance * estimated_price
        
        # Subtract estimated operational costs
        estimated_operational_cost = self._calculate_operational_costs(company) * 1.1  # 10% increase for uncertainty
        company.expected_next_hour_profit -= estimated_operational_cost
    
    def _calculate_operational_costs(self, company: Company) -> float:
        """Calculate operational costs for the company"""
        total_cost = 0.0
        
        # Battery maintenance costs
        for station in company.stations:
            for battery in station.batteries:
                # Cost increases with lower SOH
                if battery.soh < 80:
                    total_cost += (100 - battery.soh) * 0.01  # $0.01 per % below 80%
                
                # Critical SOC emergency costs
                if battery.soc < 10:
                    total_cost += 5.0  # $5 emergency cost
                elif battery.soc < 20:
                    total_cost += 1.0  # $1 warning cost
            
            # Inverter maintenance costs
            for inverter in station.inverters:
                if inverter.efficiency < 85:
                    total_cost += (90 - inverter.efficiency) * 0.1  # $0.1 per % below 85%
            
            # Solar panel maintenance costs
            for panel in station.solar_panels:
                if panel.efficiency < 80:
                    total_cost += (85 - panel.efficiency) * 0.05  # $0.05 per % below 80%
        
        return total_cost
    
    def get_profit_forecast_24h(self, company: Company) -> Tuple[List[float], List[float], List[float]]:
        """Get 24-hour profit forecast with load and generation predictions including battery benefits"""
        if not self.prediction_service:
            # Fallback forecast based on current patterns
            return self._get_fallback_profit_forecast(company)
        
        try:
            # Get 24-hour predictions for all stations
            load_forecasts = []
            generation_forecasts = []
            profit_forecasts = []
            
            # Create battery state simulation for 24 hours
            battery_states = self._initialize_battery_forecast_states(company)
            
            for hour_offset in range(24):
                total_load = 0.0
                total_generation = 0.0
                
                for station in company.stations:
                    # Get predictions (in real implementation, this would be hour-specific predictions)
                    load_24h = self.prediction_service.predict_load_24h(station.id)
                    power_24h = self.prediction_service.predict_power_24h(station.id)
                    
                    total_load += load_24h[hour_offset] if hour_offset < len(load_24h) else load_24h[-1]
                    total_generation += power_24h[hour_offset] if hour_offset < len(power_24h) else power_24h[-1]
                
                load_forecasts.append(total_load)
                generation_forecasts.append(total_generation)
                
                # Calculate expected profit for this hour including battery benefits
                hour_profit = self._calculate_hourly_profit_with_battery(
                    company, total_generation, total_load, battery_states, hour_offset
                )
                
                profit_forecasts.append(hour_profit)
            
            return profit_forecasts, load_forecasts, generation_forecasts
            
        except Exception as e:
            print(f"Error in profit forecast: {e}")
            return self._get_fallback_profit_forecast(company)
    
    def _initialize_battery_forecast_states(self, company: Company) -> dict:
        """Initialize battery states for 24-hour forecast simulation"""
        battery_states = {}
        
        for station_idx, station in enumerate(company.stations):
            station_batteries = []
            for battery_idx, battery in enumerate(station.batteries):
                # Create forecast state for each battery
                battery_state = {
                    'soc': battery.soc,  # Current state of charge
                    'capacity': battery.capacity,  # Battery capacity in kWh
                    'soh': battery.soh,  # State of health
                    'available': not battery.maintenance_required,
                    'max_charge_rate': 10.0,  # kW
                    'max_discharge_rate': 10.0,  # kW
                    'min_soc': 20.0,  # Don't discharge below 20%
                    'max_soc': 95.0   # Don't charge above 95%
                }
                station_batteries.append(battery_state)
            battery_states[station_idx] = station_batteries
        
        return battery_states
    
    def _calculate_hourly_profit_with_battery(self, company: Company, generation: float, load: float, 
                                            battery_states: dict, hour_offset: int) -> float:
        """Calculate hourly profit including battery discharge benefits"""
        hour_profit = 0.0
        
        # Calculate raw energy balance
        raw_balance = generation - load
        
        # Simulate battery management for this hour
        total_battery_discharge = 0.0
        total_battery_charge = 0.0
        
        if raw_balance < 0:  # Deficit - use batteries to discharge
            deficit = abs(raw_balance)
            remaining_deficit = deficit
            
            # Try to cover deficit with battery discharge
            for station_idx in battery_states:
                if remaining_deficit <= 0:
                    break
                    
                station_batteries = battery_states[station_idx]
                # Sort by SOC (highest first for discharge)
                available_batteries = [(i, b) for i, b in enumerate(station_batteries) 
                                     if b['available'] and b['soc'] > b['min_soc']]
                available_batteries.sort(key=lambda x: -x[1]['soc'])
                
                for battery_idx, battery_state in available_batteries:
                    if remaining_deficit <= 0:
                        break
                        
                    # Calculate how much this battery can discharge
                    available_energy = (battery_state['soc'] - battery_state['min_soc']) / 100.0 * battery_state['capacity']
                    max_discharge_this_hour = min(
                        battery_state['max_discharge_rate'],  # Rate limit
                        available_energy,  # Energy limit
                        remaining_deficit  # Need limit
                    )
                    
                    if max_discharge_this_hour > 0.1:  # Minimum discharge threshold
                        total_battery_discharge += max_discharge_this_hour
                        remaining_deficit -= max_discharge_this_hour
                        
                        # Update battery state for next hour
                        energy_used = max_discharge_this_hour
                        soc_reduction = (energy_used / battery_state['capacity']) * 100.0
                        battery_state['soc'] = max(battery_state['min_soc'], 
                                                 battery_state['soc'] - soc_reduction)
            
            # Calculate profit from battery discharge (avoided grid purchases)
            if total_battery_discharge > 0:
                hour_profit += total_battery_discharge * self.grid_price
            
            # Calculate cost for remaining deficit (grid purchase)
            if remaining_deficit > 0:
                hour_profit -= remaining_deficit * self.grid_price * 1.05  # Premium for grid purchase
        
        elif raw_balance > 0:  # Surplus - charge batteries and sell excess
            surplus = raw_balance
            remaining_surplus = surplus
            
            # Try to charge batteries with surplus
            for station_idx in battery_states:
                if remaining_surplus <= 0:
                    break
                    
                station_batteries = battery_states[station_idx]
                # Sort by SOC (lowest first for charging)
                available_batteries = [(i, b) for i, b in enumerate(station_batteries) 
                                     if b['available'] and b['soc'] < b['max_soc']]
                available_batteries.sort(key=lambda x: x[1]['soc'])
                
                for battery_idx, battery_state in available_batteries:
                    if remaining_surplus <= 0:
                        break
                        
                    # Calculate how much this battery can charge
                    available_capacity = (battery_state['max_soc'] - battery_state['soc']) / 100.0 * battery_state['capacity']
                    max_charge_this_hour = min(
                        battery_state['max_charge_rate'],  # Rate limit
                        available_capacity,  # Capacity limit
                        remaining_surplus  # Available energy limit
                    )
                    
                    if max_charge_this_hour > 0.1:  # Minimum charge threshold
                        total_battery_charge += max_charge_this_hour
                        remaining_surplus -= max_charge_this_hour
                        
                        # Update battery state for next hour
                        energy_stored = max_charge_this_hour * 0.9  # 90% charging efficiency
                        soc_increase = (energy_stored / battery_state['capacity']) * 100.0
                        battery_state['soc'] = min(battery_state['max_soc'], 
                                                 battery_state['soc'] + soc_increase)
            
            # Calculate profit from selling remaining surplus
            if remaining_surplus > 0:
                hour_profit += remaining_surplus * self.grid_price * 0.95  # Discount for selling
        
        # Subtract operational costs
        operational_cost = self._calculate_operational_costs(company) / 24.0  # Hourly portion
        hour_profit -= operational_cost
        
        return hour_profit
    
    def _get_fallback_profit_forecast(self, company: Company) -> Tuple[List[float], List[float], List[float]]:
        """Fallback profit forecast when prediction service is unavailable"""
        current_hour = datetime.now().hour
        profit_forecasts = []
        load_forecasts = []
        generation_forecasts = []
        
        # Base on current company performance with some variation
        current_total_load = sum(station.current_load for station in company.stations)
        current_total_generation = sum(station.current_generation for station in company.stations)
        
        # Initialize battery states for fallback forecast
        battery_states = self._initialize_battery_forecast_states(company)
        
        for hour_offset in range(24):
            hour = (current_hour + hour_offset) % 24
            
            # Simulate load pattern
            if 8 <= hour <= 18:  # Business hours
                load_multiplier = 1.2 + 0.3 * np.sin((hour - 8) * np.pi / 10)
            elif 18 <= hour <= 22:  # Evening peak
                load_multiplier = 1.5 + 0.2 * np.sin((hour - 18) * np.pi / 4)
            else:  # Night/early morning
                load_multiplier = 0.7 + 0.1 * np.sin(hour * np.pi / 12)
            
            # Simulate generation pattern (solar)
            if 6 <= hour <= 18:  # Daylight hours
                solar_angle = (hour - 12) / 6
                generation_multiplier = np.exp(-2 * solar_angle**2)  # Gaussian curve
            else:
                generation_multiplier = 0.0
            
            hour_load = current_total_load * load_multiplier * (1 + np.random.normal(0, 0.1))
            hour_generation = current_total_generation * generation_multiplier * (1 + np.random.normal(0, 0.1))
            
            load_forecasts.append(max(0, hour_load))
            generation_forecasts.append(max(0, hour_generation))
            
            # Calculate profit with battery benefits
            hour_profit = self._calculate_hourly_profit_with_battery(
                company, hour_generation, hour_load, battery_states, hour_offset
            )
            profit_forecasts.append(hour_profit)
        
        return profit_forecasts, load_forecasts, generation_forecasts
    
    def _rule_based_energy_management(self):
        """Rule-based energy management logic"""
        for company in self.companies:
            for station in company.stations:
                # Battery management rules
                self._rule_based_battery_management(station)
                
                # Energy trading rules
                if company.total_surplus > 0:
                    self._execute_energy_sale(company)
                elif company.total_deficit > 0:
                    self._execute_energy_purchase(company)
    
    def _rule_based_battery_management(self, station: ChargingStation):
        """Rule-based battery charging/discharging logic"""
        # Sort batteries by SOC (ascending) and SOH (descending)
        available_batteries = [b for b in station.batteries if not b.maintenance_required]
        available_batteries.sort(key=lambda x: (x.soc, -x.soh))
        
        available_energy = station.current_generation - station.current_load
        
        if available_energy > 0:  # Surplus energy
            # Charge battery with lowest SOC first
            for battery in available_batteries:
                if battery.soc < 90 and available_energy > 0:
                    charge_power = min(available_energy, 10)  # Max 10kW charging
                    battery.current_power = charge_power
                    available_energy -= charge_power
                    if available_energy <= 0:
                        break
        
        elif available_energy < 0:  # Energy deficit
            # Discharge batteries with highest SOC first
            available_batteries.sort(key=lambda x: (-x.soc, -x.soh))
            energy_needed = abs(available_energy)
            
            for battery in available_batteries:
                if battery.soc > 20 and energy_needed > 0:
                    discharge_power = min(energy_needed, 10)  # Max 10kW discharging
                    battery.current_power = -discharge_power
                    energy_needed -= discharge_power
                    if energy_needed <= 0:
                        break
    
    def _rl_energy_management(self):
        """RL-based energy management using advanced agent"""
        # First, manage batteries for all companies using rule-based approach
        for company in self.companies:
            self._manage_battery_storage(company)
            # Recalculate energy balance after battery management
            self._calculate_energy_balance(company)
        
        # Then apply RL for trading decisions
        if self.rl_agent and RL_AVAILABLE:
            self._advanced_rl_management()
        else:
            self._simple_rl_management()
    
    def _advanced_rl_management(self):
        """Advanced RL-based energy management"""
        current_hour = datetime.now().hour
        
        for company in self.companies:
            # Get current state
            previous_balance = company.total_surplus - company.total_deficit
            
            # Choose action using hybrid agent
            action = self.rl_agent.choose_action(company, self.companies, self.grid_price, current_hour)
            
            # Execute action
            reward = self._execute_rl_action_advanced(company, action, previous_balance, current_hour)
            
            # Get next state for learning
            next_state = self.rl_agent.rl_agent.get_state_representation(
                company, self.companies, self.grid_price, current_hour
            )
            
            # Update RL agent (simplified - in practice you'd store more state info)
            if hasattr(self, '_last_states') and company.id in self._last_states:
                last_state = self._last_states[company.id]
                last_action = self._last_actions.get(company.id, 0)
                
                self.rl_agent.update_rl_agent(
                    last_state, last_action, reward, next_state, False
                )
            
            # Store current state for next iteration
            if not hasattr(self, '_last_states'):
                self._last_states = {}
                self._last_actions = {}
            
            self._last_states[company.id] = next_state
            self._last_actions[company.id] = action
        
        # Periodically save the model
        if random.random() < 0.1:  # 10% chance to save each update
            self.rl_agent.save_model()
    
    def _simple_rl_management(self):
        """Simple RL-based energy management (fallback)"""
        for company in self.companies:
            state = self._get_rl_state(company)
            action = self._select_rl_action(state)
            reward = self._execute_rl_action(company, action)
            self._update_q_table(state, action, reward)
    
    def _execute_rl_action_advanced(self, company, action, previous_balance, hour):
        """Execute advanced RL action and return reward"""
        action_names = [
            "hold", "sell_to_grid", "buy_from_grid",
            "sell_to_company_1", "sell_to_company_2", 
            "buy_from_company_1", "buy_from_company_2",
            "emergency_charge"
        ]
        
        reward = 0.0
        
        if action == 0:  # Hold
            reward = 1.0
            
        elif action == 1:  # Sell to grid
            if company.total_surplus > 0:
                revenue = company.total_surplus * self.grid_price
                reward = revenue * 100
                self._log_trade(company.name, "Grid", company.total_surplus, self.grid_price)
            else:
                reward = -50
                
        elif action == 2:  # Buy from grid
            if company.total_deficit > 0:
                cost = company.total_deficit * self.grid_price
                reward = -cost * 50 + 20
                self._log_trade("Grid", company.name, company.total_deficit, self.grid_price)
            else:
                reward = -30
                
        elif action in [3, 4]:  # Sell to other companies
            if company.total_surplus > 0:
                other_companies = [c for c in self.companies if c.id != company.id]
                company_index = action - 3
                if company_index < len(other_companies):
                    buyer = other_companies[company_index]
                    if buyer.total_deficit > 0:
                        trade_amount = min(company.total_surplus, buyer.total_deficit)
                        revenue = trade_amount * buyer.energy_price
                        reward = revenue * 120
                        
                        if buyer.energy_price > self.grid_price:
                            reward += (buyer.energy_price - self.grid_price) * trade_amount * 50
                        
                        self._log_trade(company.name, buyer.name, trade_amount, buyer.energy_price)
                    else:
                        reward = -40
            else:
                reward = -50
                
        elif action in [5, 6]:  # Buy from other companies
            if company.total_deficit > 0:
                other_companies = [c for c in self.companies if c.id != company.id]
                company_index = action - 5
                if company_index < len(other_companies):
                    seller = other_companies[company_index]
                    if seller.total_surplus > 0:
                        trade_amount = min(company.total_deficit, seller.total_surplus)
                        cost = trade_amount * seller.energy_price
                        reward = -cost * 40 + 25
                        
                        if seller.energy_price < self.grid_price:
                            reward += (self.grid_price - seller.energy_price) * trade_amount * 60
                        
                        self._log_trade(seller.name, company.name, trade_amount, seller.energy_price)
                    else:
                        reward = -40
            else:
                reward = -30
                
        elif action == 7:  # Emergency charge
            critical_batteries = sum(
                len([b for b in s.batteries if b.soc < 10])
                for s in company.stations
            )
            
            if critical_batteries > 0:
                reward = 100 * critical_batteries
                # Execute emergency charging from cheapest source
                if company.total_deficit > 0:
                    self._log_trade("Grid", company.name, company.total_deficit, self.grid_price)
            else:
                reward = -20
        
        # Calculate reward using the advanced agent's reward function if available
        if self.rl_agent and RL_AVAILABLE:
            detailed_reward = self.rl_agent.rl_agent.calculate_reward(
                company, action, self.companies, self.grid_price, previous_balance
            )
            reward = detailed_reward
        
        return reward
    
    def _get_rl_state(self, company: Company) -> Tuple:
        """Get current state for RL algorithm"""
        other_companies = [c for c in self.companies if c.id != company.id]
        neighbor_prices = [c.energy_price for c in other_companies]
        
        return (
            round(self.grid_price, 3),
            round(company.total_surplus - company.total_deficit, 1),
            tuple(round(p, 3) for p in neighbor_prices)
        )
    
    def _select_rl_action(self, state: Tuple) -> str:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: random action
            actions = ["sell_grid", "sell_company", "buy_grid", "buy_company", "hold"]
            return random.choice(actions)
        else:
            # Exploit: best known action
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return "hold"
    
    def _execute_rl_action(self, company: Company, action: str) -> float:
        """Execute RL action and return reward"""
        reward = 0.0
        
        if action == "sell_grid" and company.total_surplus > 0:
            revenue = company.total_surplus * self.grid_price
            reward = revenue
            self._log_trade(company.name, "Grid", company.total_surplus, self.grid_price)
            
        elif action == "sell_company" and company.total_surplus > 0:
            # Find best buyer
            buyers = [c for c in self.companies if c.id != company.id and c.total_deficit > 0]
            if buyers:
                best_buyer = max(buyers, key=lambda x: x.energy_price)
                trade_amount = min(company.total_surplus, best_buyer.total_deficit)
                revenue = trade_amount * best_buyer.energy_price
                reward = revenue
                self._log_trade(company.name, best_buyer.name, trade_amount, best_buyer.energy_price)
        
        return reward
    
    def _update_q_table(self, state: Tuple, action: str, reward: float):
        """Update Q-table with new experience"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ["sell_grid", "sell_company", "buy_grid", "buy_company", "hold"]}
        
        # Simple Q-learning update
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (reward - old_value)
    
    def _execute_energy_sale(self, company: Company):
        """Execute energy sale decision"""
        if company.total_surplus <= 0:
            return
        
        # Find best price among grid and other companies
        grid_revenue = company.total_surplus * self.grid_price
        
        best_company_buyer = None
        best_company_revenue = 0
        
        for other_company in self.companies:
            if other_company.id != company.id and other_company.total_deficit > 0:
                trade_amount = min(company.total_surplus, other_company.total_deficit)
                revenue = trade_amount * other_company.energy_price
                if revenue > best_company_revenue:
                    best_company_revenue = revenue
                    best_company_buyer = other_company
        
        # Choose best option
        if best_company_revenue > grid_revenue and best_company_buyer:
            trade_amount = min(company.total_surplus, best_company_buyer.total_deficit)
            self._log_trade(company.name, best_company_buyer.name, trade_amount, best_company_buyer.energy_price)
        else:
            self._log_trade(company.name, "Grid", company.total_surplus, self.grid_price)
    
    def _execute_energy_purchase(self, company: Company):
        """Execute energy purchase decision"""
        if company.total_deficit <= 0:
            return
        
        # Find cheapest source
        cheapest_price = self.grid_price
        cheapest_source = "Grid"
        
        for other_company in self.companies:
            if other_company.id != company.id and other_company.total_surplus > 0:
                if other_company.energy_price < cheapest_price:
                    cheapest_price = other_company.energy_price
                    cheapest_source = other_company.name
        
        self._log_trade(cheapest_source, company.name, company.total_deficit, cheapest_price)
    
    def _log_trade(self, seller: str, buyer: str, amount: float, price: float):
        """Log trading transaction"""
        trade = {
            "timestamp": datetime.now(),
            "seller": seller,
            "buyer": buyer,
            "amount_kwh": round(amount, 2),
            "price_per_kwh": round(price, 4),
            "total_value": round(amount * price, 2)
        }
        self.trading_log.append(trade)
    
    def _check_maintenance_alerts(self):
        """Check for maintenance requirements"""
        self.maintenance_alerts = []
        
        for company in self.companies:
            for station in company.stations:
                # Check batteries
                for battery in station.batteries:
                    if battery.soc < self.battery_critical_soc:
                        alert = {
                            "type": "CRITICAL",
                            "component": "Battery",
                            "id": battery.id,
                            "station": station.name,
                            "company": company.name,
                            "message": f"Battery SOC critical: {battery.soc:.1f}%",
                            "timestamp": datetime.now()
                        }
                        self.maintenance_alerts.append(alert)
                        battery.maintenance_required = True
                    
                    elif battery.soc < self.battery_low_soc:
                        alert = {
                            "type": "WARNING",
                            "component": "Battery",
                            "id": battery.id,
                            "station": station.name,
                            "company": company.name,
                            "message": f"Battery SOC low: {battery.soc:.1f}%",
                            "timestamp": datetime.now()
                        }
                        self.maintenance_alerts.append(alert)
                    
                    if battery.soh < 70:
                        alert = {
                            "type": "MAINTENANCE",
                            "component": "Battery",
                            "id": battery.id,
                            "station": station.name,
                            "company": company.name,
                            "message": f"Battery SOH low: {battery.soh:.1f}%",
                            "timestamp": datetime.now()
                        }
                        self.maintenance_alerts.append(alert)
                        battery.maintenance_required = True
                
                # Check inverters
                for inverter in station.inverters:
                    if inverter.efficiency < self.inverter_efficiency_threshold:
                        alert = {
                            "type": "MAINTENANCE",
                            "component": "Inverter",
                            "id": inverter.id,
                            "station": station.name,
                            "company": company.name,
                            "message": f"Inverter efficiency low: {inverter.efficiency:.1f}%",
                            "timestamp": datetime.now()
                        }
                        self.maintenance_alerts.append(alert)
                        inverter.maintenance_required = True
                
                # Check solar panels
                for panel in station.solar_panels:
                    if panel.efficiency < self.solar_efficiency_threshold:
                        alert = {
                            "type": "MAINTENANCE",
                            "component": "Solar Panel",
                            "id": panel.id,
                            "station": station.name,
                            "company": company.name,
                            "message": f"Solar panel efficiency low: {panel.efficiency:.1f}%",
                            "timestamp": datetime.now()
                        }
                        self.maintenance_alerts.append(alert)
                        panel.maintenance_required = True
    
    def _manage_battery_storage(self, company: Company):
        """Rule-based battery management for energy storage optimization and EV charging"""
        current_hour = datetime.now().hour
        
        for station in company.stations:
            station_surplus = station.current_generation - station.current_load
            ev_charging_demand = station.current_load * 0.8  # Assume 80% of load is EV charging
            
            # Calculate total available battery energy for EV charging
            available_battery_energy = sum(
                max(0, (battery.soc - 20) / 100 * battery.capacity) 
                for battery in station.batteries 
                if not battery.maintenance_required
            )
            
            # Rule-based battery charging/discharging decisions
            for battery in station.batteries:
                if battery.maintenance_required:
                    continue  # Skip batteries that need maintenance
                
                # Priority Rule: Use batteries to support EV charging when solar is insufficient
                if station.current_generation < ev_charging_demand and battery.soc > 25:
                    # Calculate how much this battery should contribute to EV charging
                    battery_capacity_ratio = battery.capacity / sum(b.capacity for b in station.batteries if not b.maintenance_required)
                    ev_support_needed = (ev_charging_demand - station.current_generation) * battery_capacity_ratio
                    discharge_rate = min(ev_support_needed, battery.capacity * 0.2, 20.0)  # Max 20% capacity or 20kW
                    
                    if discharge_rate > 0:
                        battery.current_power = -discharge_rate
                        battery.soc = max(20, battery.soc - (discharge_rate / battery.capacity) * 100)
                        continue
                
                # Rule 1: Emergency discharge - very low grid supply hours (peak demand)
                elif current_hour in [18, 19, 20, 21]:  # Peak evening hours
                    if battery.soc > 30:  # Only discharge if sufficient charge
                        discharge_rate = min(15.0, battery.capacity * 0.1)  # 10% of capacity max
                        battery.current_power = -discharge_rate
                        battery.soc = max(20, battery.soc - (discharge_rate / battery.capacity) * 100)
                
                # Rule 2: Store energy during high generation hours
                elif current_hour in [10, 11, 12, 13, 14, 15]:  # High solar generation hours
                    if station_surplus > 0 and battery.soc < 90:  # Surplus available and battery not full
                        charge_rate = min(station_surplus * 0.3, battery.capacity * 0.15)  # Use 30% of surplus, max 15% of capacity
                        battery.current_power = charge_rate
                        battery.soc = min(100, battery.soc + (charge_rate / battery.capacity) * 100)
                
                # Rule 3: Maintain minimum charge during night hours
                elif current_hour in [22, 23, 0, 1, 2, 3]:  # Night hours
                    if battery.soc < 25:  # Low charge, need to preserve
                        charge_rate = min(5.0, battery.capacity * 0.05)  # Gentle charging
                        battery.current_power = charge_rate
                        battery.soc = min(100, battery.soc + (charge_rate / battery.capacity) * 100)
                    else:
                        battery.current_power = 0  # Idle
                
                # Rule 4: Discharge during moderate demand hours if well charged
                elif current_hour in [16, 17]:  # Pre-peak hours
                    if battery.soc > 60:  # Well charged
                        discharge_rate = min(8.0, battery.capacity * 0.08)
                        battery.current_power = -discharge_rate
                        battery.soc = max(20, battery.soc - (discharge_rate / battery.capacity) * 100)
                
                # Rule 5: Default behavior - maintain optimal charge range
                else:
                    if battery.soc < 40:  # Below optimal range
                        charge_rate = min(3.0, battery.capacity * 0.03)
                        battery.current_power = charge_rate
                        battery.soc = min(100, battery.soc + (charge_rate / battery.capacity) * 100)
                    elif battery.soc > 85:  # Above optimal range, can discharge
                        discharge_rate = min(2.0, battery.capacity * 0.02)
                        battery.current_power = -discharge_rate
                        battery.soc = max(20, battery.soc - (discharge_rate / battery.capacity) * 100)
                    else:
                        battery.current_power = 0  # Optimal range, maintain
    
    def get_company_battery_status(self, company: Company) -> Dict:
        """Get comprehensive battery status for a company"""
        total_capacity = 0
        total_stored_energy = 0
        total_charging_power = 0
        total_discharging_power = 0
        battery_count = 0
        batteries_by_station = {}
        
        for station in company.stations:
            station_data = {
                'total_capacity': 0,
                'total_stored': 0,
                'charging_power': 0,
                'discharging_power': 0,
                'batteries': []
            }
            
            for battery in station.batteries:
                battery_count += 1
                total_capacity += battery.capacity
                stored_energy = (battery.soc / 100) * battery.capacity
                total_stored_energy += stored_energy
                
                station_data['total_capacity'] += battery.capacity
                station_data['total_stored'] += stored_energy
                
                if battery.current_power > 0:  # Charging
                    total_charging_power += battery.current_power
                    station_data['charging_power'] += battery.current_power
                elif battery.current_power < 0:  # Discharging
                    total_discharging_power += abs(battery.current_power)
                    station_data['discharging_power'] += abs(battery.current_power)
                
                station_data['batteries'].append({
                    'id': battery.id,
                    'soc': battery.soc,
                    'soh': battery.soh,
                    'capacity': battery.capacity,
                    'stored_energy': stored_energy,
                    'current_power': battery.current_power,
                    'status': 'charging' if battery.current_power > 0 else ('discharging' if battery.current_power < 0 else 'idle')
                })
            
            batteries_by_station[station.id] = station_data
        
        return {
            'total_capacity_kwh': total_capacity,
            'total_stored_kwh': total_stored_energy,
            'total_soc_percent': (total_stored_energy / total_capacity * 100) if total_capacity > 0 else 0,
            'total_charging_power_kw': total_charging_power,
            'total_discharging_power_kw': total_discharging_power,
            'net_battery_power_kw': total_discharging_power - total_charging_power,  # Positive = discharging to grid
            'battery_count': battery_count,
            'stations': batteries_by_station
        }
    
    def _rule_based_energy_management(self):
        """Enhanced rule-based energy management with battery optimization"""
        for company in self.companies:
            # First, optimize battery storage
            self._manage_battery_storage(company)
            
            # Then recalculate energy balance with updated battery states
            self._calculate_energy_balance(company)
            
            # Execute trading decisions
            self._execute_trading_rules(company)
    
    def _execute_trading_rules(self, company: Company):
        """Execute trading rules based on energy balance"""
        current_hour = datetime.now().hour
        
        # Rule 1: If surplus, try to sell to other companies first (better price)
        if company.total_surplus > 0:
            for other_company in self.companies:
                if other_company.id != company.id and other_company.total_deficit > 0:
                    trade_amount = min(company.total_surplus, other_company.total_deficit)
                    if other_company.energy_price > self.grid_price * 0.9:  # Only if price is reasonable
                        self._log_trade(company.name, other_company.name, trade_amount, other_company.energy_price)
                        company.total_surplus -= trade_amount
                        other_company.total_deficit -= trade_amount
                        if company.total_surplus <= 0:
                            break
            
            # Sell remaining surplus to grid
            if company.total_surplus > 0:
                self._log_trade(company.name, "Grid", company.total_surplus, self.grid_price)
        
        # Rule 2: If deficit, try to buy from other companies first (cheaper price)
        elif company.total_deficit > 0:
            for other_company in self.companies:
                if other_company.id != company.id and other_company.total_surplus > 0:
                    trade_amount = min(company.total_deficit, other_company.total_surplus)
                    if other_company.energy_price < self.grid_price * 1.1:  # Only if price is reasonable
                        self._log_trade(other_company.name, company.name, trade_amount, other_company.energy_price)
                        company.total_deficit -= trade_amount
                        other_company.total_surplus -= trade_amount
                        if company.total_deficit <= 0:
                            break
            
            # Buy remaining deficit from grid
            if company.total_deficit > 0:
                self._log_trade("Grid", company.name, company.total_deficit, self.grid_price)

# Energy Management System is ready to be imported and used