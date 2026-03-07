# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:47:04 2026

@author: clher
"""
# %% Import and setup
from ultracold.analysis.fitting import fit_gaussian_1d
from ultracold.analysis.profiles import extract_horizontal_profile
import numpy as np
import matplotlib.pyplot as plt
from ultracold.simulation.cloud import UltracoldCloud

# %% Create cloud
cloud = UltracoldCloud(
    n_atoms=1e6,
    temperature=100e-9,
    trap_frequencies=(2*np.pi*20, 2*np.pi*20, 2*np.pi*100)
)

print(f"Cloud sizes: {cloud.thermal_sizes}")

# %% Generate image
x = np.linspace(-500e-6, 500e-6, 200)
y = np.linspace(-500e-6, 500e-6, 200)
X, Y = np.meshgrid(x, y)

image = cloud.generate_image(X, Y, noise_level=0.02)

# %% Plot
plt.figure(figsize=(8, 6))
plt.imshow(image, extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6])
plt.colorbar(label='Column density')
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title('Simulated thermal cloud')
plt.show()

# %% Fit profile

x_px, profile = extract_horizontal_profile(image, average_rows=5)
result = fit_gaussian_1d(x_px, profile)
print(result)
