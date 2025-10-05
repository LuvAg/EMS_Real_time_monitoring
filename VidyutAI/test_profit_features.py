"""
Test script to verify profit calculation features
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from energy_management_system import EnergyManagementSystem

def test_profit_features():
    print("🧪 Testing Profit Calculation Features")
    print("=" * 50)
    
    # Initialize EMS
    ems = EnergyManagementSystem()
    
    # Update system to calculate profits
    print("🔄 Updating system state to calculate profits...")
    ems.update_system_state()
    
    print("\n💰 Company Profit Analysis:")
    print("-" * 30)
    
    for company in ems.companies:
        print(f"\n📊 {company.name}:")
        print(f"  Current Hour Profit: ${company.current_hour_profit:.2f}")
        print(f"  Expected Next Hour: ${company.expected_next_hour_profit:.2f}")
        print(f"  Cumulative Profit: ${company.cumulative_profit:.2f}")
        print(f"  Profit History Length: {len(company.profit_history)}")
        
        if company.profit_history:
            print(f"  Latest Profit: ${company.profit_history[-1]:.2f}")
        
        # Get 24-hour forecast
        print(f"\n📈 24-Hour Forecast for {company.name}:")
        try:
            profit_forecast, load_forecast, gen_forecast = ems.get_profit_forecast_24h(company)
            
            print(f"  Total 24h Forecast Profit: ${sum(profit_forecast):.2f}")
            print(f"  Average Hourly Profit: ${sum(profit_forecast)/24:.2f}")
            print(f"  Best Hour Profit: ${max(profit_forecast):.2f}")
            print(f"  Worst Hour Loss: ${min(profit_forecast):.2f}")
            
            # Show first few hours
            print(f"  Next 6 Hours Forecast:")
            for i in range(min(6, len(profit_forecast))):
                status = "Surplus" if gen_forecast[i] >= load_forecast[i] else "Deficit"
                balance = gen_forecast[i] - load_forecast[i]
                print(f"    Hour {i+1}: Load={load_forecast[i]:.1f}kW, Gen={gen_forecast[i]:.1f}kW, "
                      f"Balance={balance:.1f}kW ({status}), Profit=${profit_forecast[i]:.2f}")
                
        except Exception as e:
            print(f"  ❌ Error getting forecast: {e}")
    
    print("\n🏢 System-wide Profit Summary:")
    print("-" * 30)
    total_current_profit = sum(c.current_hour_profit for c in ems.companies)
    total_expected_profit = sum(c.expected_next_hour_profit for c in ems.companies)
    total_cumulative = sum(c.cumulative_profit for c in ems.companies)
    
    print(f"Total Current Hour Profit: ${total_current_profit:.2f}")
    print(f"Total Expected Next Hour: ${total_expected_profit:.2f}")
    print(f"Total Cumulative Profit: ${total_cumulative:.2f}")
    
    # Test another system update to see profit evolution
    print(f"\n🔄 Testing profit evolution with second update...")
    ems.update_system_state()
    
    print("Updated Profits:")
    for company in ems.companies:
        print(f"  {company.name}: Current=${company.current_hour_profit:.2f}, "
              f"Cumulative=${company.cumulative_profit:.2f}, "
              f"History Length={len(company.profit_history)}")

if __name__ == "__main__":
    test_profit_features()
    print("\n✅ Profit testing completed!")
    print("Check the dashboard at: http://localhost:8503")
    print("Navigate to 'Your Company' page to see profit metrics")
    print("Navigate to 'Profit Forecast' page to see 24-hour forecasts")