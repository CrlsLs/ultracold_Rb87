# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 18:05:12 2026

@author: clher
"""
#!/usr/bin/env python3
"""Quick test to verify the simulation setup works."""


# Add src to path
import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("Testing ultracold atoms simulation setup...\n")

# Test imports
print("1. Testing imports...")
try:
    from ultracold.utils.constants import M_RB87, STATE_F2_MF2, ExperimentParams
    from ultracold.simulation.cloud import ThermalCloud, MixedStateCloud
    from ultracold.simulation.free_fall import simulate_free_fall, create_experiment_cloud
    from ultracold.analysis.profiles import extract_horizontal_profile, profile_from_simulation
    from ultracold.analysis.cloud_analysis import analyze_cloud_image
    print("   ✅ All imports successful!")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test cloud creation
print("\n2. Testing cloud creation...")

cloud = create_experiment_cloud()
print(f"   ✅ Created: {cloud}")

# Test simulation
print("\n3. Testing free fall simulation...")
times = np.linspace(0, 20e-3, 50)
result = simulate_free_fall(cloud, times)
print(f"   ✅ Simulation complete: {result.n_times} time points")

# Test profile extraction
print("\n4. Testing profile extraction...")
profile = profile_from_simulation(cloud, 'z', n_points=100, extent=300e-6)
print(f"   ✅ Profile extracted: {len(profile.values)} points")

# Test image generation
print("\n5. Testing image generation...")
x = np.linspace(-300e-6, 300e-6, 100)
y = np.linspace(-300e-6, 300e-6, 100)
X, Y = np.meshgrid(x, y)

cloud.release()
cloud.evolve(10e-3)
image = cloud.generate_image(X, Y, noise_level=0.01)
print(f"   ✅ Image generated: shape = {image.shape}")

# Test analysis
print("\n6. Testing cloud analysis...")
params = analyze_cloud_image(image, pixel_size=6e-6)
print(f"   ✅ Analysis complete: {params}")

print("\n" + "="*50)
print(" ALL TESTS PASSED!")
print("="*50)
print("\nYou can now run the full simulation with:")
print("  python scripts/simulate_free_fall.py")
