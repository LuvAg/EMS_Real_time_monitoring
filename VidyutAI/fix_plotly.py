#!/usr/bin/env python3
"""Fix Plotly deprecation warnings in dashboard"""

def fix_plotly_warnings():
    with open('energy_dashboard.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace width='stretch' with proper config
    content = content.replace(
        "width='stretch'", 
        "config={'displayModeBar': False}, use_container_width=True"
    )
    
    with open('energy_dashboard.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed all Plotly deprecation warnings")

if __name__ == "__main__":
    fix_plotly_warnings()