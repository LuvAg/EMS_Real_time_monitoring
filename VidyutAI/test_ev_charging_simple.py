#!/usr/bin/env python3
"""
Simple test script to verify EV charging with battery support
"""

import sys
import os
from energy_management_system import EnergyManagementSystem

def test_ev_battery_integration():
    """Test that batteries are integrated with EV charging"""
    print("🔋⚡ Testing EV Charging with Battery Integration")
    print("=" * 60)
    
    # Initialize system
    ems = EnergyManagementSystem()
    
    print(f"✅ Energy Management System initialized")
    print(f"🚗 Analyzing EV charging with battery support...")
    print()
    
    # Update system state
    ems.update_system_state()
    
    # Analyze each company
    for company in ems.companies:
        print(f"🏢 {company.name} - EV Charging Analysis")
        print("-" * 40)
        
        # Company-wide metrics
        total_load = sum(station.current_load for station in company.stations)
        ev_load = total_load * 0.8  # 80% is EV charging
        total_generation = sum(station.current_generation for station in company.stations)
        
        # Battery metrics
        battery_status = ems.get_company_battery_status(company)
        battery_discharging = battery_status['total_discharging_power_kw']
        battery_charging = battery_status['total_charging_power_kw']
        
        print(f"  Total Load: {total_load:.1f} kW")
        print(f"  EV Charging Load: {ev_load:.1f} kW ({ev_load/total_load*100:.1f}% of total)")
        print(f"  Solar Generation: {total_generation:.1f} kW")
        print(f"  Battery Discharging: {battery_discharging:.1f} kW")
        print(f"  Battery Charging: {battery_charging:.1f} kW")
        
        # Calculate energy flow to EVs
        solar_to_evs = min(total_generation, ev_load)
        battery_to_evs = min(battery_discharging, max(0, ev_load - solar_to_evs))
        grid_to_evs = max(0, ev_load - solar_to_evs - battery_to_evs)
        
        print(f"  Solar → EVs: {solar_to_evs:.1f} kW")
        print(f"  Battery → EVs: {battery_to_evs:.1f} kW")
        print(f"  Grid → EVs: {grid_to_evs:.1f} kW")
        
        # Calculate renewable percentage
        renewable_support = (solar_to_evs + battery_to_evs) / ev_load * 100 if ev_load > 0 else 0
        print(f"  Renewable EV Support: {renewable_support:.1f}%")
        
        # Battery status
        print(f"  Average Battery SOC: {battery_status['total_soc_percent']:.1f}%")
        print(f"  Total Battery Capacity: {battery_status['total_capacity_kwh']:.1f} kWh")
        print(f"  Stored Energy: {battery_status['total_stored_kwh']:.1f} kWh")
        
        # Check if batteries are actively supporting EV charging
        if battery_to_evs > 0:
            print(f"  ✅ Batteries are supporting EV charging!")
        elif battery_discharging > 0:
            print(f"  ⚡ Batteries are discharging (may support EVs)")
        else:
            print(f"  💤 Batteries not currently discharging")
        
        print()
    
    # Station-level analysis
    print("🔍 Station-Level EV Charging Analysis")
    print("-" * 40)
    
    for company in ems.companies:
        for station in company.stations:
            station_ev_load = station.current_load * 0.8
            station_generation = station.current_generation
            
            # Calculate station battery contribution
            station_battery_discharge = sum(
                abs(battery.current_power) 
                for battery in station.batteries 
                if battery.current_power < 0
            )
            
            station_battery_charge = sum(
                battery.current_power 
                for battery in station.batteries 
                if battery.current_power > 0
            )
            
            print(f"{company.name} - {station.name}:")
            print(f"  EV Load: {station_ev_load:.1f} kW")
            print(f"  Solar Gen: {station_generation:.1f} kW")
            print(f"  Battery Out: {station_battery_discharge:.1f} kW")
            print(f"  Battery In: {station_battery_charge:.1f} kW")
            
            # Check energy balance
            if station_generation < station_ev_load and station_battery_discharge > 0:
                shortfall = station_ev_load - station_generation
                battery_coverage = min(station_battery_discharge / shortfall, 1.0) * 100
                print(f"  ⚡ Energy shortfall: {shortfall:.1f} kW")
                print(f"  🔋 Battery coverage: {battery_coverage:.1f}%")
                print(f"  ✅ Batteries helping with EV charging!")
            elif station_generation > station_ev_load and station_battery_charge > 0:
                surplus = station_generation - station_ev_load
                print(f"  ☀️ Energy surplus: {surplus:.1f} kW")
                print(f"  🔋 Storing {station_battery_charge:.1f} kW for later EV charging")
            else:
                print(f"  ⚖️ Energy balanced or batteries idle")
            
            print()
    
    print("✅ EV charging and battery integration analysis completed!")
    print("🌐 Dashboard available at: http://localhost:8503")
    print("📊 Check 'Your Company' page for detailed EV charging metrics")

if __name__ == "__main__":
    try:
        test_ev_battery_integration()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)