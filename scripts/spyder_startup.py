# -*- coding: utf-8 -*-

# Save as: ultracold_atoms_project/scripts/spyder_startup.py

"""
Spyder startup script - runs when IPython console starts.
Configure in: Tools > Preferences > IPython console > Startup
"""

import sys
import os
from pathlib import Path

# Set project root (adjust path as needed)
PROJECT_ROOT = Path(r"C:\Users\clher\Documents\programacion\projects\ultracold_Rb87")

# Change to project directory
os.chdir(PROJECT_ROOT)

# Add src to path
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Common imports for interactive work
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Your package imports
try:
    from ultracold.simulation.cloud import UltracoldCloud, BECCloud
    from ultracold.analysis import fitting, profiles
    from ultracold.visualization.plots import plot_cloud_image
    print("✅ Ultracold package loaded successfully!")
except ImportError as e:
    print(f"⚠️ Could not import ultracold package: {e}")

# Configure matplotlib for interactive use
plt.ion()  # Interactive mode

# Print confirmation
print(f"📂 Working directory: {os.getcwd()}")
print(f"🐍 Python {sys.version.split()[0]}")
print("\nReady to work! Common imports available:")
print("  np, plt, pd, UltracoldCloud, BECCloud, fitting, profiles")
