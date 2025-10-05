"""
Test script to verify energy prediction integration
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from energy_prediction_service import EnergyPredictionService
from energy_management_system import EnergyManagementSystem

def test_prediction_service():
    print("🧪 Testing Energy Prediction Service")
    print("=" * 50)
    
    # Initialize prediction service
    service = EnergyPredictionService()
    
    # Check availability
    availability = service.is_available()
    print("Service Availability:")
    for key, value in availability.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    print("\n📊 Testing Load Prediction:")
    try:
        load_24h = service.predict_load_24h()
        print(f"  - 24-hour load predictions shape: {load_24h.shape}")
        print(f"  - Current hour load: {load_24h[0]:.2f} kWh")
        print(f"  - Next hour load: {load_24h[1]:.2f} kWh")
        print(f"  - Peak load (24h): {load_24h.max():.2f} kWh")
        print(f"  - Min load (24h): {load_24h.min():.2f} kWh")
        print(f"  - Average load (24h): {load_24h.mean():.2f} kWh")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n☀️ Testing Power Prediction:")
    try:
        power_24h = service.predict_power_24h()
        print(f"  - 24-hour power predictions shape: {power_24h.shape}")
        print(f"  - Current hour generation: {power_24h[0]:.2f} kW")
        print(f"  - Next hour generation: {power_24h[1]:.2f} kW")
        print(f"  - Peak generation (24h): {power_24h.max():.2f} kW")
        print(f"  - Min generation (24h): {power_24h.min():.2f} kW")
        print(f"  - Average generation (24h): {power_24h.mean():.2f} kW")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n⚡ Testing Current Predictions:")
    try:
        current_load, current_gen = service.get_current_predictions()
        next_load, next_gen = service.get_next_hour_predictions()
        print(f"  - Current: Load={current_load:.2f} kW, Generation={current_gen:.2f} kW")
        print(f"  - Next Hour: Load={next_load:.2f} kW, Generation={next_gen:.2f} kW")
        
        balance_current = current_gen - current_load
        balance_next = next_gen - next_load
        print(f"  - Current Balance: {balance_current:.2f} kW ({'Surplus' if balance_current > 0 else 'Deficit'})")
        print(f"  - Next Balance: {balance_next:.2f} kW ({'Surplus' if balance_next > 0 else 'Deficit'})")
    except Exception as e:
        print(f"  ❌ Error: {e}")

def test_energy_management_system():
    print("\n\n🏢 Testing Energy Management System Integration")
    print("=" * 50)
    
    # Initialize EMS
    ems = EnergyManagementSystem()
    
    print("System Status:")
    print(f"  - Companies: {len(ems.companies)}")
    print(f"  - Total Stations: {sum(len(c.stations) for c in ems.companies)}")
    print(f"  - Prediction Service Available: {'✅' if ems.prediction_service else '❌'}")
    print(f"  - RL Agent Available: {'✅' if ems.rl_agent else '❌'}")
    
    print("\nCompany Initial State:")
    for company in ems.companies:
        total_load = sum(s.current_load for s in company.stations)
        total_gen = sum(s.current_generation for s in company.stations)
        balance = total_gen - total_load
        
        print(f"  📊 {company.name}:")
        print(f"    - Load: {total_load:.2f} kW")
        print(f"    - Generation: {total_gen:.2f} kW")
        print(f"    - Balance: {balance:.2f} kW ({'Surplus' if balance > 0 else 'Deficit'})")
        print(f"    - Price: ${company.energy_price:.4f}/kWh")
    
    print("\n🔄 Testing System Update:")
    try:
        ems.update_system_state()
        print("  ✅ System state updated successfully")
        
        print("\nPost-Update State:")
        for company in ems.companies:
            total_load = sum(s.current_load for s in company.stations)
            total_gen = sum(s.current_generation for s in company.stations)
            
            print(f"  📊 {company.name}:")
            print(f"    - Surplus: {company.total_surplus:.2f} kW")
            print(f"    - Deficit: {company.total_deficit:.2f} kW")
            print(f"    - Updated Price: ${company.energy_price:.4f}/kWh")
        
        if ems.maintenance_alerts:
            print(f"\n🚨 Maintenance Alerts: {len(ems.maintenance_alerts)}")
            for alert in ems.maintenance_alerts[:3]:  # Show first 3
                print(f"    - {alert['type']}: {alert['message']}")
        
        if ems.trading_log:
            print(f"\n💰 Trading Activity: {len(ems.trading_log)} transactions")
            for trade in ems.trading_log[-3:]:  # Show last 3
                print(f"    - {trade['seller']} → {trade['buyer']}: {trade['amount_kwh']} kWh at ${trade['price_per_kwh']:.4f}")
        
    except Exception as e:
        print(f"  ❌ Error updating system: {e}")

def test_station_details():
    print("\n\n🏭 Testing Station Details")
    print("=" * 50)
    
    ems = EnergyManagementSystem()
    
    # Test first station of first company
    station = ems.companies[0].stations[0]
    print(f"Station: {station.name} ({station.id})")
    print(f"Location: {station.location}")
    print(f"Current Load: {station.current_load:.2f} kW")
    print(f"Current Generation: {station.current_generation:.2f} kW")
    print(f"Predicted Load: {station.predicted_load:.2f} kW")
    print(f"Predicted Generation: {station.predicted_generation:.2f} kW")
    
    print(f"\n🔋 Batteries ({len(station.batteries)}):")
    for i, battery in enumerate(station.batteries[:3]):  # Show first 3
        print(f"  Battery {i+1}: SOC={battery.soc:.1f}%, SOH={battery.soh:.1f}%, Power={battery.current_power:.1f}kW")
    
    print(f"\n⚡ Inverters ({len(station.inverters)}):")
    for i, inverter in enumerate(station.inverters):
        print(f"  Inverter {i+1}: Efficiency={inverter.efficiency:.1f}%, Load={inverter.current_load:.1f}kW")
    
    print(f"\n☀️ Solar Panels ({len(station.solar_panels)}):")
    total_panel_gen = sum(p.current_generation for p in station.solar_panels)
    for i, panel in enumerate(station.solar_panels[:3]):  # Show first 3
        print(f"  Panel {i+1}: Efficiency={panel.efficiency:.1f}%, Gen={panel.current_generation:.1f}kW")
    print(f"  Total Panel Generation: {total_panel_gen:.2f} kW")
    print(f"  Station Generation: {station.current_generation:.2f} kW")
    print(f"  Scaling Factor Applied: {station.current_generation/total_panel_gen if total_panel_gen > 0 else 'N/A'}")

if __name__ == "__main__":
    print("🚀 Energy Management System Integration Test")
    print("=" * 60)
    
    test_prediction_service()
    test_energy_management_system()
    test_station_details()
    
    print("\n" + "=" * 60)
    print("✅ Integration testing completed!")
    print("You can now run the dashboard with: streamlit run energy_dashboard.py")