#!/usr/bin/env python3
"""
Test script to verify EV charging with battery support
"""

import sys
import os
from datetime import datetime
from energy_management_system import EnergyManagementSystem

def test_ev_charging_with_batteries():
    """Test that batteries are being used to charge EVs"""
    print("🔋⚡ Testing EV Charging with Battery Support")
    print("=" * 60)
    
    # Initialize system
    ems = EnergyManagementSystem()
    
    print(f"✅ Energy Management System initialized")
    print(f"🚗 Testing EV charging scenarios...")
    print()
    
    # Test different scenarios
    scenarios = [
        {"hour": 8, "description": "Morning charging (low solar)"},
        {"hour": 12, "description": "Midday charging (high solar)"},
        {"hour": 18, "description": "Evening charging (peak demand)"},
        {"hour": 22, "description": "Night charging (no solar)"}
    ]
    
    for scenario in scenarios:
        print(f"📊 Scenario: {scenario['description']} (Hour {scenario['hour']})")
        print("-" * 50)
        
        # Simulate different time
        import datetime as dt
        original_now = dt.datetime.now
        dt.datetime.now = lambda: original_now().replace(hour=scenario['hour'])
        
        # Update system state
        ems.update_system_state()
        
        # Analyze each company
        for company in ems.companies:
            company_name = company.name
            
            # Calculate metrics
            total_load = sum(station.current_load for station in company.stations)
            ev_load = total_load * 0.8  # 80% is EV charging
            total_generation = sum(station.current_generation for station in company.stations)
            
            # Calculate battery contribution
            battery_status = ems.get_company_battery_status(company)
            battery_to_evs = max(0, battery_status['total_discharging_power_kw'])
            solar_to_evs = min(total_generation, ev_load)
            grid_to_evs = max(0, ev_load - solar_to_evs - battery_to_evs)
            
            print(f"  {company_name}:")
            print(f"    EV Charging Demand: {ev_load:.1f} kW")
            print(f"    Solar Generation: {total_generation:.1f} kW")
            print(f"    Battery → EVs: {battery_to_evs:.1f} kW")
            print(f"    Solar → EVs: {solar_to_evs:.1f} kW")
            print(f"    Grid → EVs: {grid_to_evs:.1f} kW")
            
            # Calculate efficiency
            renewable_support = (battery_to_evs + solar_to_evs) / ev_load * 100 if ev_load > 0 else 0
            print(f"    Renewable Support: {renewable_support:.1f}%")
            
            # Battery status
            avg_soc = battery_status['total_soc_percent']
            charging_count = sum(
                sum(1 for battery in station.batteries if battery.current_power > 0)
                for station in company.stations
            )
            discharging_count = sum(
                sum(1 for battery in station.batteries if battery.current_power < 0)
                for station in company.stations
            )
            
            print(f"    Avg Battery SOC: {avg_soc:.1f}%")
            print(f"    Charging Batteries: {charging_count}")
            print(f"    Discharging Batteries: {discharging_count}")
            print()
        
        # Restore original datetime
        dt.datetime.now = original_now
        print()
    
    # Test battery discharge priority for EV charging
    print("🎯 Testing Battery Discharge Priority for EV Charging")
    print("-" * 50)
    
    for company in ems.companies:
        for station in company.stations:
            ev_demand = station.current_load * 0.8
            solar_available = station.current_generation
            
            if solar_available < ev_demand:  # Solar insufficient for EV charging
                shortfall = ev_demand - solar_available
                battery_contribution = 0
                
                for battery in station.batteries:
                    if battery.current_power < 0:  # Discharging
                        battery_contribution += abs(battery.current_power)
                
                print(f"  {company.name} - {station.name}:")
                print(f"    EV Demand: {ev_demand:.1f} kW")
                print(f"    Solar Available: {solar_available:.1f} kW")
                print(f"    Energy Shortfall: {shortfall:.1f} kW")
                print(f"    Battery Support: {battery_contribution:.1f} kW")
                
                if battery_contribution > 0:
                    support_ratio = min(battery_contribution / shortfall * 100, 100)
                    print(f"    Battery Coverage: {support_ratio:.1f}% of shortfall")
                    print(f"    ✅ Batteries supporting EV charging")
                else:
                    print(f"    ⚠️  No battery support for EV charging")
                print()
    
    print("✅ EV charging with battery support testing completed!")
    print("🌐 Check dashboard at: http://localhost:8503")
    print("📊 Navigate to 'Your Company' page to see EV charging metrics")

if __name__ == "__main__":
    try:
        test_ev_charging_with_batteries()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)