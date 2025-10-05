#!/usr/bin/env python3
"""
Test to demonstrate battery discharge profit with lower operational costs
"""

import sys
sys.path.append('.')

from energy_management_system import EnergyManagementSystem

def test_battery_profit_with_good_equipment():
    """Test battery profit with well-maintained equipment (lower operational costs)"""
    print("🔋 Testing Battery Discharge Profit - Optimized Equipment")
    print("=" * 60)
    
    ems = EnergyManagementSystem()
    company = ems.companies[0]
    
    # Improve equipment condition to reduce operational costs
    for station in company.stations:
        # Set batteries to good condition
        for battery in station.batteries:
            battery.soh = 95.0  # High SOH
            battery.soc = 80.0  # High SOC for discharge
        
        # Set inverters to high efficiency
        for inverter in station.inverters:
            inverter.efficiency = 95.0  # High efficiency
        
        # Set solar panels to high efficiency  
        for panel in station.solar_panels:
            panel.efficiency = 90.0  # High efficiency
        
        # Set low generation, high load scenario
        station.current_generation = 5.0
        station.current_load = 25.0
    
    print(f"Grid Price: ${ems.grid_price:.3f} per kWh")
    
    # Check operational costs with improved equipment
    operational_cost = ems._calculate_operational_costs(company)
    print(f"Operational Costs (optimized equipment): ${operational_cost:.2f}")
    
    # Run system update
    print(f"\n⚡ Running system with optimized equipment...")
    ems.update_system_state()
    
    # Analyze results
    total_battery_discharge = sum(
        sum(max(0, -battery.current_power) for battery in station.batteries)
        for station in company.stations
    )
    
    battery_profit = total_battery_discharge * ems.grid_price
    grid_cost = company.total_deficit * ems.grid_price
    actual_operational_cost = ems._calculate_operational_costs(company)
    expected_profit = battery_profit - grid_cost - actual_operational_cost
    actual_profit = company.current_hour_profit
    
    print(f"\n📊 RESULTS WITH OPTIMIZED EQUIPMENT:")
    print(f"Battery Discharge: {total_battery_discharge:.2f} kWh")
    print(f"Battery Profit: ${battery_profit:.2f}")
    print(f"Grid Cost: ${grid_cost:.2f}")
    print(f"Operational Cost: ${actual_operational_cost:.2f}")
    print(f"Expected Profit: ${expected_profit:.2f}")
    print(f"Actual Profit: ${actual_profit:.2f}")
    
    print(f"\n🎯 COMPARISON:")
    if actual_profit > 0:
        print("✅ NET PROFIT achieved with optimized equipment!")
        print("✅ Battery discharge provides economic benefit")
    else:
        print("⚠️  Still net loss, but battery reduces losses")
        
    savings_without_battery = (company.total_deficit + total_battery_discharge) * ems.grid_price + actual_operational_cost
    print(f"\nWithout battery (all from grid): ${savings_without_battery:.2f} cost")
    print(f"With battery discharge: ${abs(actual_profit):.2f} {'profit' if actual_profit > 0 else 'loss'}")
    savings = savings_without_battery - abs(actual_profit)
    print(f"Battery saves: ${savings:.2f}")

if __name__ == "__main__":
    test_battery_profit_with_good_equipment()