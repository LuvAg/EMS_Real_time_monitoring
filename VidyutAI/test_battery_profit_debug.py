#!/usr/bin/env python3
"""
Enhanced test script with detailed profit calculation debugging
"""

import sys
sys.path.append('.')

from energy_management_system import EnergyManagementSystem, OperationMode
import time

def test_battery_discharge_profit_debug():
    """Test battery discharge profit with detailed debugging"""
    print("🔋 Testing Battery Discharge Profit - DEBUG MODE")
    print("=" * 60)
    
    # Initialize system
    ems = EnergyManagementSystem()
    
    # Get first company for testing
    company = ems.companies[0]
    initial_profit = company.cumulative_profit
    
    print(f"Initial Company Profit: ${initial_profit:.2f}")
    print(f"Grid Price: ${ems.grid_price:.3f} per kWh")
    
    # Force a scenario where we have high load and low generation
    for station in company.stations:
        station.current_generation = 5.0  # Low generation
        station.current_load = 25.0  # High load requiring battery support
        
        # Ensure batteries have charge to discharge
        for battery in station.batteries:
            battery.soc = 80.0  # High SOC for discharge
    
    print(f"\nBefore simulation:")
    print(f"- Setup Generation: 5.0 kWh per station × {len(company.stations)} stations")
    print(f"- Setup Load: 25.0 kWh per station × {len(company.stations)} stations")
    
    # Capture profit before update
    profit_before = company.current_hour_profit
    cumulative_before = company.cumulative_profit
    
    print(f"\nProfit before update: ${profit_before:.2f}")
    print(f"Cumulative before update: ${cumulative_before:.2f}")
    
    # Run the system update
    print(f"\n⚡ Running system update...")
    ems.update_system_state()
    
    # Now analyze what happened step by step
    print(f"\n📊 DETAILED ANALYSIS:")
    
    # 1. Calculate actual battery discharge
    total_battery_discharge = sum(
        sum(max(0, -battery.current_power) for battery in station.batteries)
        for station in company.stations
    )
    print(f"1. Total Battery Discharge: {total_battery_discharge:.2f} kWh")
    
    # 2. Calculate expected profit from battery
    battery_profit = total_battery_discharge * ems.grid_price
    print(f"2. Expected Battery Profit: ${battery_profit:.2f} ({total_battery_discharge:.2f} × ${ems.grid_price:.3f})")
    
    # 3. Check company energy balance
    total_generation = sum(station.current_generation for station in company.stations)
    total_load = sum(station.current_load for station in company.stations)
    print(f"3. Total Generation: {total_generation:.2f} kWh")
    print(f"   Total Load: {total_load:.2f} kWh")
    print(f"   Raw Deficit: {total_load - total_generation:.2f} kWh")
    
    # 4. Available energy (generation + battery discharge)
    available_energy = total_generation + total_battery_discharge
    print(f"4. Available Energy: {available_energy:.2f} kWh (gen + battery)")
    
    # 5. Net balance after battery
    net_balance = available_energy - total_load
    print(f"5. Net Balance: {net_balance:.2f} kWh")
    
    # 6. Company surplus/deficit as calculated
    surplus = company.total_surplus
    deficit = company.total_deficit
    print(f"6. Company Surplus: {surplus:.2f} kWh")
    print(f"   Company Deficit: {deficit:.2f} kWh")
    
    # 7. Grid purchase cost
    grid_cost = deficit * ems.grid_price if deficit > 0 else 0
    print(f"7. Grid Purchase Cost: ${grid_cost:.2f} ({deficit:.2f} × ${ems.grid_price:.3f})")
    
    # 8. Operational costs
    operational_cost = ems._calculate_operational_costs(company)
    print(f"8. Operational Costs: ${operational_cost:.2f}")
    
    # 9. Expected total profit
    expected_profit = battery_profit - grid_cost - operational_cost
    print(f"9. Expected Total Profit: ${expected_profit:.2f} (battery - grid - operational)")
    
    # 10. Actual calculated profit
    actual_profit = company.current_hour_profit
    actual_cumulative = company.cumulative_profit
    print(f"10. Actual Profit: ${actual_profit:.2f}")
    print(f"    Cumulative Profit: ${actual_cumulative:.2f}")
    
    # 11. Discrepancy analysis
    discrepancy = actual_profit - expected_profit
    print(f"11. Discrepancy: ${discrepancy:.2f}")
    
    print(f"\n🎯 SUMMARY:")
    if abs(discrepancy) < 0.01:
        print("✅ PROFIT CALCULATION CORRECT")
    else:
        print("❌ PROFIT CALCULATION ISSUE DETECTED")
        print(f"   Expected: ${expected_profit:.2f}")
        print(f"   Actual: ${actual_profit:.2f}")
        print(f"   Difference: ${discrepancy:.2f}")
    
    if actual_profit > 0:
        print("✅ Battery discharge generates NET PROFIT")
    else:
        print("❌ Battery discharge results in NET LOSS")
        if battery_profit > abs(actual_profit):
            print("   ✓ Battery savings exceed other costs")
        else:
            print("   ❌ Other costs exceed battery savings")

if __name__ == "__main__":
    test_battery_discharge_profit_debug()