#!/usr/bin/env python3
"""
Test to analyze operational costs breakdown
"""

import sys
sys.path.append('.')

from energy_management_system import EnergyManagementSystem

def analyze_operational_costs():
    """Analyze what's driving high operational costs"""
    print("💰 Analyzing Operational Costs Breakdown")
    print("=" * 50)
    
    ems = EnergyManagementSystem()
    company = ems.companies[0]
    
    print(f"Company: {company.id}")
    print(f"Stations: {len(company.stations)}")
    
    total_cost = 0.0
    
    for i, station in enumerate(company.stations):
        print(f"\n🏭 Station {i+1}:")
        station_cost = 0.0
        
        # Battery costs
        print(f"  Batteries ({len(station.batteries)}):")
        for j, battery in enumerate(station.batteries):
            battery_cost = 0.0
            
            # SOH costs
            if battery.soh < 80:
                soh_cost = (100 - battery.soh) * 0.01
                battery_cost += soh_cost
                print(f"    Battery {j+1}: SOH={battery.soh:.1f}% → ${soh_cost:.2f} (low SOH penalty)")
            
            # SOC costs  
            if battery.soc < 10:
                soc_cost = 5.0
                battery_cost += soc_cost
                print(f"    Battery {j+1}: SOC={battery.soc:.1f}% → ${soc_cost:.2f} (critical SOC emergency)")
            elif battery.soc < 20:
                soc_cost = 1.0
                battery_cost += soc_cost
                print(f"    Battery {j+1}: SOC={battery.soc:.1f}% → ${soc_cost:.2f} (low SOC warning)")
            
            if battery_cost == 0:
                print(f"    Battery {j+1}: SOH={battery.soh:.1f}%, SOC={battery.soc:.1f}% → $0.00 (good condition)")
            
            station_cost += battery_cost
        
        # Inverter costs
        print(f"  Inverters ({len(station.inverters)}):")
        for j, inverter in enumerate(station.inverters):
            inverter_cost = 0.0
            if inverter.efficiency < 85:
                inverter_cost = (90 - inverter.efficiency) * 0.1
                station_cost += inverter_cost
                print(f"    Inverter {j+1}: Eff={inverter.efficiency:.1f}% → ${inverter_cost:.2f} (low efficiency)")
            else:
                print(f"    Inverter {j+1}: Eff={inverter.efficiency:.1f}% → $0.00 (good efficiency)")
        
        # Solar panel costs
        print(f"  Solar Panels ({len(station.solar_panels)}):")
        for j, panel in enumerate(station.solar_panels):
            panel_cost = 0.0
            if panel.efficiency < 80:
                panel_cost = (85 - panel.efficiency) * 0.05
                station_cost += panel_cost
                print(f"    Panel {j+1}: Eff={panel.efficiency:.1f}% → ${panel_cost:.2f} (low efficiency)")
            else:
                print(f"    Panel {j+1}: Eff={panel.efficiency:.1f}% → $0.00 (good efficiency)")
        
        print(f"  Station {i+1} Total Cost: ${station_cost:.2f}")
        total_cost += station_cost
    
    print(f"\n💰 TOTAL OPERATIONAL COST: ${total_cost:.2f}")
    
    # Verify with actual calculation
    actual_cost = ems._calculate_operational_costs(company)
    print(f"   Actual calculated cost: ${actual_cost:.2f}")
    
    if abs(total_cost - actual_cost) < 0.01:
        print("✅ Manual calculation matches system calculation")
    else:
        print(f"❌ Discrepancy: ${abs(total_cost - actual_cost):.2f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if total_cost > 5.0:
        print("❌ Operational costs are very high!")
        print("   Consider reducing penalty rates for better system economics")
    elif total_cost > 2.0:
        print("⚠️  Operational costs are moderate")
    else:
        print("✅ Operational costs are reasonable")

if __name__ == "__main__":
    analyze_operational_costs()