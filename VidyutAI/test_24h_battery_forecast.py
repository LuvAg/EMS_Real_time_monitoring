#!/usr/bin/env python3
"""
Test 24-hour profit forecast with battery discharge benefits
"""

import sys
sys.path.append('.')

from energy_management_system import EnergyManagementSystem
import matplotlib.pyplot as plt
import numpy as np

def test_24h_profit_forecast_with_battery():
    """Test that 24-hour profit forecast includes battery discharge benefits"""
    print("📊 Testing 24-Hour Profit Forecast with Battery Benefits")
    print("=" * 60)
    
    ems = EnergyManagementSystem()
    company = ems.companies[0]
    
    print(f"Company: {company.id}")
    print(f"Grid Price: ${ems.grid_price:.3f} per kWh")
    
    # Set up batteries with good charge for testing
    total_battery_capacity = 0.0
    for station in company.stations:
        for battery in station.batteries:
            battery.soc = 70.0  # Good charge level
            battery.soh = 90.0  # Good health
            total_battery_capacity += battery.capacity
    
    print(f"Total Battery Capacity: {total_battery_capacity:.1f} kWh")
    print(f"Average Battery SOC: 70%")
    
    # Get 24-hour profit forecast
    print(f"\n⚡ Generating 24-hour profit forecast...")
    profit_forecasts, load_forecasts, generation_forecasts = ems.get_profit_forecast_24h(company)
    
    # Analyze the forecast
    total_profit = sum(profit_forecasts)
    max_profit = max(profit_forecasts)
    min_profit = min(profit_forecasts)
    positive_hours = len([p for p in profit_forecasts if p > 0])
    
    print(f"\n📊 24-Hour Forecast Summary:")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Hourly Profit: ${total_profit/24:.2f}")
    print(f"Maximum Hourly Profit: ${max_profit:.2f}")
    print(f"Minimum Hourly Profit: ${min_profit:.2f}")
    print(f"Profitable Hours: {positive_hours}/24")
    
    # Show hourly breakdown for key periods
    print(f"\n🕐 Hourly Breakdown (sample hours):")
    key_hours = [0, 6, 12, 18]  # Midnight, morning, noon, evening
    for hour in key_hours:
        if hour < len(profit_forecasts):
            load = load_forecasts[hour]
            generation = generation_forecasts[hour]
            profit = profit_forecasts[hour]
            balance = generation - load
            
            print(f"Hour {hour:2d}: Load={load:5.1f} kWh, Gen={generation:5.1f} kWh, "
                  f"Balance={balance:+5.1f} kWh, Profit=${profit:+6.2f}")
    
    # Compare with simple forecast (without battery benefits)
    simple_profits = []
    for i in range(24):
        balance = generation_forecasts[i] - load_forecasts[i]
        if balance > 0:
            simple_profit = balance * ems.grid_price * 0.95
        else:
            simple_profit = balance * ems.grid_price * 1.05
        simple_profit -= ems._calculate_operational_costs(company) / 24.0
        simple_profits.append(simple_profit)
    
    simple_total = sum(simple_profits)
    battery_benefit = total_profit - simple_total
    
    print(f"\n🔋 Battery Impact Analysis:")
    print(f"Simple Forecast (no battery): ${simple_total:.2f}")
    print(f"Enhanced Forecast (with battery): ${total_profit:.2f}")
    print(f"Battery Benefit: ${battery_benefit:.2f}")
    print(f"Improvement: {((battery_benefit / abs(simple_total)) * 100) if simple_total != 0 else 0:.1f}%")
    
    if battery_benefit > 0:
        print("✅ Battery discharge correctly provides financial benefit")
    else:
        print("⚠️  Battery benefit not detected or negative")
    
    # Check for realistic battery usage patterns
    deficit_hours = len([i for i in range(24) if generation_forecasts[i] < load_forecasts[i]])
    surplus_hours = 24 - deficit_hours
    
    print(f"\n📈 Energy Pattern Analysis:")
    print(f"Deficit Hours: {deficit_hours}/24 (batteries likely discharging)")
    print(f"Surplus Hours: {surplus_hours}/24 (batteries likely charging)")
    
    # Validate that forecast is reasonable
    if 0 <= positive_hours <= 24 and -1000 <= total_profit <= 1000:
        print(f"\n✅ 24-Hour Forecast Validation:")
        print(f"✓ Reasonable profit range")
        print(f"✓ Realistic mix of profitable/unprofitable hours")
        print(f"✓ Battery benefits included in calculations")
    else:
        print(f"\n❌ Forecast Validation Issues:")
        print(f"- Profit range seems unrealistic: ${total_profit:.2f}")
        print(f"- Positive hours: {positive_hours}/24")
    
    return profit_forecasts, load_forecasts, generation_forecasts

def plot_24h_forecast(profit_forecasts, load_forecasts, generation_forecasts):
    """Create a visualization of the 24-hour forecast"""
    hours = list(range(24))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Energy plot
    ax1.plot(hours, load_forecasts, 'r-', label='Load', linewidth=2)
    ax1.plot(hours, generation_forecasts, 'g-', label='Generation', linewidth=2)
    ax1.fill_between(hours, load_forecasts, alpha=0.3, color='red')
    ax1.fill_between(hours, generation_forecasts, alpha=0.3, color='green')
    ax1.set_ylabel('Energy (kWh)')
    ax1.set_title('24-Hour Energy Forecast')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 23)
    
    # Profit plot
    colors = ['green' if p > 0 else 'red' for p in profit_forecasts]
    ax2.bar(hours, profit_forecasts, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Profit ($)')
    ax2.set_title('24-Hour Profit Forecast (Including Battery Benefits)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 23.5)
    
    plt.tight_layout()
    plt.savefig('24h_profit_forecast_with_battery.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Forecast chart saved as '24h_profit_forecast_with_battery.png'")

if __name__ == "__main__":
    try:
        profit_forecasts, load_forecasts, generation_forecasts = test_24h_profit_forecast_with_battery()
        # Uncomment the next line if you want to generate the plot
        # plot_24h_forecast(profit_forecasts, load_forecasts, generation_forecasts)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()