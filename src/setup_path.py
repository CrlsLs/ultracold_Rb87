# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:28:58 2026

@author: clher
"""

# Save this as: setup_path.py in your project root
# Run it ONCE in Spyder to set up the path permanently

import sys
from pathlib import Path

# Get the src directory
project_root = Path(__file__).parent
src_path = project_root / "src"

print(f"Adding to Python path: {src_path}")

# For current session
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print("✅ Path added for current session")

# Show current path
print("\nCurrent sys.path:")
for p in sys.path[:5]:
    print(f"  {p}")