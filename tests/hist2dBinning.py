# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 23:37:02 2026

@author: clher
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 1. Generate Synthetic Data (10,000 particles)
# Creating 3 distinct clusters to make the density obvious
n = 10000
x = np.concatenate([
    np.random.normal(0, 1, int(n*0.5)),   # Center cluster
    np.random.normal(3, 0.5, int(n*0.3)),  # Side cluster
    np.random.normal(-2, 1, int(n*0.2))   # Spread out cluster
])
y = np.concatenate([
    np.random.normal(0, 1, int(n*0.5)),
    np.random.normal(3, 0.5, int(n*0.3)),
    np.random.normal(-2, 1, int(n*0.2))
])

# Setup the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# -------------------------------------------------------
# Method A: 2D Histogram (plt.hist2d)
# -------------------------------------------------------
# 'bins' controls the resolution (100x100 pixels)
# 'cmap' controls the color scheme (e.g., 'inferno', 'viridis', 'plasma')
h = axes[0].hist2d(x, y, bins=100, cmap='inferno')
axes[0].set_title("2D Histogram (Fast)")
plt.colorbar(h[3], ax=axes[0], label='Count')

# -------------------------------------------------------
# Method B: Hexbin (plt.hexbin)
# -------------------------------------------------------
# 'gridsize' controls the number of hexagons
hb = axes[1].hexbin(x, y, gridsize=50, cmap='inferno', mincnt=1)
axes[1].set_title("Hexagonal Binning")
plt.colorbar(hb, ax=axes[1], label='Count')

# -------------------------------------------------------
# Method C: Gaussian KDE (Smooth)
# -------------------------------------------------------
# Calculate the point density
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so the densest points are plotted last (on top)
idx = z.argsort()
x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]

# Scatter plot colored by density
sc = axes[2].scatter(x_sorted, y_sorted, c=z_sorted,
                     s=5, cmap='inferno', edgecolor='')
axes[2].set_title("Gaussian KDE (Smooth)")
plt.colorbar(sc, ax=axes[2], label='Density')

plt.tight_layout()
plt.show()
