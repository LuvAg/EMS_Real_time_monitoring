#!/usr/bin/env python3
"""
Test to show current battery capacities in the system
"""

import sys
sys.path.append('.')

from energy_management_system import EnergyManagementSystem

def show_battery_capacities():
    """Display current battery capacity distribution"""
    print("🔋 Current Battery Capacity Analysis")
    print("=" * 50)
    
    ems = EnergyManagementSystem()
    
    all_capacities = []
    total_capacity = 0.0
    
    for company in ems.companies:
        company_capacity = 0.0
        print(f"\n📊 {company.id}:")
        
        for i, station in enumerate(company.stations):
            station_capacity = 0.0
            print(f"  Station {i+1}:")
            
            for j, battery in enumerate(station.batteries):
                all_capacities.append(battery.capacity)
                station_capacity += battery.capacity
                company_capacity += battery.capacity
                total_capacity += battery.capacity
                print(f"    Battery {j+1}: {battery.capacity:.1f} kWh")
            
            print(f"    Station Total: {station_capacity:.1f} kWh")
        
        print(f"  Company Total: {company_capacity:.1f} kWh")
    
    print(f"\n📈 SYSTEM SUMMARY:")
    print(f"Total System Capacity: {total_capacity:.1f} kWh")
    print(f"Number of Batteries: {len(all_capacities)}")
    print(f"Average Battery Capacity: {sum(all_capacities)/len(all_capacities):.1f} kWh")
    print(f"Minimum Battery Capacity: {min(all_capacities):.1f} kWh")
    print(f"Maximum Battery Capacity: {max(all_capacities):.1f} kWh")
    
    # Show capacity distribution
    ranges = [
        (0, 75, "Small (≤75 kWh)"),
        (75, 125, "Medium (75-125 kWh)"),
        (125, 175, "Large (125-175 kWh)"),
        (175, 250, "Extra Large (≥175 kWh)")
    ]
    
    print(f"\n📊 CAPACITY DISTRIBUTION:")
    for min_cap, max_cap, label in ranges:
        count = len([c for c in all_capacities if min_cap < c <= max_cap])
        percentage = (count / len(all_capacities)) * 100
        print(f"{label}: {count} batteries ({percentage:.1f}%)")
    
    # Check if maximum is over 200 kWh
    over_200 = [c for c in all_capacities if c > 200]
    if over_200:
        print(f"\n⚠️  BATTERIES OVER 200 kWh:")
        for i, cap in enumerate(over_200):
            print(f"  Battery {i+1}: {cap:.1f} kWh")
        print(f"  Total over 200 kWh: {len(over_200)} batteries")
    else:
        print(f"\n✅ All batteries are ≤200 kWh (current maximum)")

if __name__ == "__main__":
    show_battery_capacities()