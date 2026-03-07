# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:27:50 2026

@author: clher
"""

# Save as: scripts/verify_setup.py

"""
Verify that the project is set up correctly.
Run this in Spyder to check everything works.
"""


import os
import sys
from pathlib import Path
print("=" * 50)
print("PROJECT SETUP VERIFICATION")
print("=" * 50)

# Check working directory
cwd = Path(os.getcwd())
print(f"\n📂 Current directory: {cwd}")

# Check if we're in the project
if (cwd / "src" / "ultracold").exists():
    print("   ✅ In project root")
else:
    print("   ⚠️ Not in project root!")
    print(f"   Expected to find: {cwd / 'src' / 'ultracold'}")

# Check Python path
print(f"\n🐍 Python path includes src/:")
src_in_path = any("src" in p for p in sys.path)
print(f"   {'✅ Yes' if src_in_path else '❌ No - imports will fail!'}")

# Try imports
print("\n📦 Testing imports:")

imports_to_test = [
    ("numpy", "np"),
    ("scipy", "scipy"),
    ("matplotlib.pyplot", "plt"),
    ("pandas", "pd"),
    ("ultracold", "ultracold"),
    ("ultracold.simulation.cloud", "cloud"),
    ("ultracold.analysis.fitting", "fitting"),
    ("ultracold.analysis.profiles", "profiles"),
    ("ultracold.visualization.plots", "plots"),
]

for module, alias in imports_to_test:
    try:
        exec(f"import {module} as {alias}")
        print(f"   ✅ {module}")
    except ImportError as e:
        print(f"   ❌ {module}: {e}")

# Check directories exist
print("\n📁 Checking directories:")
required_dirs = [
    "src/ultracold",
    "scripts",
    "tests",
    "data/raw",
    "data/processed",
    "output/figures",
    "output/simulations",
    "configs",
]

for dir_path in required_dirs:
    full_path = cwd / dir_path
    if full_path.exists():
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ❌ {dir_path} (missing!)")

# Quick functionality test
print("\n🧪 Quick functionality test:")
try:
    import numpy as np
    from ultracold.simulation.cloud import UltracoldCloud

    cloud = UltracoldCloud(
        n_atoms=1e6,
        temperature=100e-9,
        trap_frequencies=(2*np.pi*20, 2*np.pi*20, 2*np.pi*100)
    )

    sizes = cloud.thermal_sizes
    print(f"   Created test cloud with sizes: {
          [f'{s*1e6:.1f} µm' for s in sizes]}")
    print("   ✅ Simulation module works!")

except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)
