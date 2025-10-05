#!/usr/bin/env python3
"""
Test script to verify that battery discharge is correctly counted as profit
when meeting load demand instead of purchasing from grid.
"""

import sys
sys.path.append('.')

from energy_management_system import EnergyManagementSystem, OperationMode
import time

def test_battery_discharge_profit():
    """Test that battery discharge generates profit by avoiding grid purchases"""
    print("🔋 Testing Battery Discharge Profit Calculation...")
    print("=" * 60)
    
    # Initialize system
    ems = EnergyManagementSystem()
    
    # Get first company for testing
    company = ems.companies[0]
    initial_profit = company.cumulative_profit
    
    print(f"Initial Company Profit: ${initial_profit:.2f}")
    print(f"Grid Price: ${ems.grid_price:.3f} per kWh")
    
    # Get battery status before simulation
    station = company.stations[0]
    battery = station.batteries[0]
    initial_soc = battery.soc
    
    print(f"\nBattery Initial SOC: {initial_soc:.1f}%")
    print(f"Battery Capacity: {battery.capacity:.1f} kWh")
    
    # Force a scenario where we have high load and low generation
    # This should trigger battery discharge
    for station in company.stations:
        # Set low generation (solar/wind)
        station.current_generation = 5.0  # Low generation
        # Set high load to force battery usage
        station.current_load = 25.0  # High load requiring battery support
        
        # Ensure batteries have charge to discharge
        for battery in station.batteries:
            battery.soc = 80.0  # High SOC for discharge
    
    print(f"\nSimulation Setup:")
    print(f"- Low Generation: 5.0 kWh per station")
    print(f"- High Load: 25.0 kWh per station") 
    print(f"- Expected Deficit: 20.0 kWh per station")
    print(f"- Battery SOC: 80% (available for discharge)")
    
    # Run one simulation step
    print(f"\n⚡ Running simulation step...")
    ems.update_system_state()
    
    # Check results
    current_profit = company.current_hour_profit
    cumulative_profit = company.cumulative_profit
    
    # Calculate total battery discharge
    total_battery_discharge = sum(
        sum(max(0, -battery.current_power) for battery in station.batteries)
        for station in company.stations
    )
    
    # Check deficit/surplus
    deficit = company.total_deficit
    surplus = company.total_surplus
    
    print(f"\n📊 Results:")
    print(f"Current Hour Profit: ${current_profit:.2f}")
    print(f"Cumulative Profit: ${cumulative_profit:.2f}")
    print(f"Total Battery Discharge: {total_battery_discharge:.2f} kWh")
    print(f"Company Total Deficit: {deficit:.2f} kWh")
    print(f"Company Total Surplus: {surplus:.2f} kWh")
    
    # Calculate expected profit from battery discharge
    expected_battery_profit = total_battery_discharge * ems.grid_price
    print(f"\nExpected Profit from Battery Discharge: ${expected_battery_profit:.2f}")
    print(f"(Battery discharge {total_battery_discharge:.2f} kWh × Grid price ${ems.grid_price:.3f})")
    
    # Check if batteries are actually discharging
    battery_discharging = False
    for station in company.stations:
        for battery in station.batteries:
            if battery.current_power < 0:  # Negative means discharging
                battery_discharging = True
                final_soc = battery.soc
                print(f"\nBattery Status:")
                print(f"- Current Power: {battery.current_power:.2f} kW (negative = discharging)")
                print(f"- SOC: {initial_soc:.1f}% → {final_soc:.1f}%")
                break
    
    # Verify profit calculation
    print(f"\n✅ Verification:")
    if battery_discharging:
        print("✓ Batteries are discharging to meet load")
        if current_profit > 0:
            print("✓ Current hour shows PROFIT (not loss)")
            print("✓ Battery discharge is correctly counted as avoided grid purchase cost")
        else:
            print("❌ Current hour shows loss - this might indicate an issue")
    else:
        print("❌ Batteries are not discharging - check battery management logic")
    
    if total_battery_discharge > 0 and expected_battery_profit > 0:
        print(f"✓ Battery discharge profit calculated: ${expected_battery_profit:.2f}")
    
    print(f"\n🎯 Summary:")
    print(f"When batteries discharge to meet load, companies avoid buying from grid")
    print(f"This should generate profit equal to: discharge_amount × grid_price")
    print(f"Current implementation: {'✓ WORKING' if current_profit > 0 and battery_discharging else '❌ NEEDS FIX'}")

if __name__ == "__main__":
    test_battery_discharge_profit()