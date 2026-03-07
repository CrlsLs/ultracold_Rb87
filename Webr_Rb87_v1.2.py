# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 00:48:00 2026

@author: clher
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 22:07:51 2026

@author: clher
"""




from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import scipy.io
from pathlib import Path
from datetime import datetime
def safe_complex(x): # A helper function to safely parse complex strings like '(5.89e-01+1.03e-02j)'
    if pd.isna(x):
        return 0j
    if isinstance(x, str):
        return complex(x.strip(' ()').replace(' ', ''))
    return complex(x)


# --- 1. Load and Parse the Psi Matrix ---
dir_load = r'C:\Users\clher\Documents\programacion\projects\ultracold_Rb87\scripts'
folder_path = Path(
    r"C:\Users\clher\Documents\programacion\projects\ultracold_Rb87\data")
Psi_path = folder_path / 'Psi_a10_kp3.mat'

mat_data = scipy.io.loadmat(Psi_path)  # load matlab mat file

Psi_or = mat_data['Psi']  # psi original

df = pd.DataFrame(Psi_or)
# %%
Psi = df.map(safe_complex).values if hasattr(
    df, 'map') else df.applymap(safe_complex).values

# Normalize Psi so max amplitude is 1
Psi_or = Psi / np.max(np.abs(Psi))  # psi original, to be cropped (or not)

# %% Crop Psi
m, n = Psi_or.shape
crf = .35  # .477  # crop factor #to crop psi original
mc = round(m*crf)
nc = round(n*crf)
Psi = Psi_or[mc:-mc, nc:-nc]  # psi cropped
print(Psi.shape)
# %%
# --- 2. Experimental Constants ---
dx = 0.8e-6  # 0.8 microns
dy = 0.8e-6
hbar = constants.hbar
c = constants.c
m_Rb87 = 1.443160648e-25  # kg

lambda_0 = 780.241e-9      # m (Rb87 D2 transition)
lambda_L = 780.368e-9      # m (Weber beam from paper)
Gamma = 2 * np.pi * 6.06e6  # rad/s
g_factor = 19.1            # a=5 mode irradiance

omega_0 = 2 * np.pi * c / lambda_0
omega_L = 2 * np.pi * c / lambda_L
delta_omega = omega_L - omega_0  # ~ -3.93e11 rad/s

g = g_factor * Gamma
p_0 = (2 * g**2) / ((Gamma / 2)**2 + delta_omega**2)

# --- 3. Field Gradients ---
I = np.abs(Psi)**2

# np.gradient returns (d/dy, d/dx) for a 2D array
dI_dy, dI_dx = np.gradient(I, dy, dx)
dPsi_dy, dPsi_dx = np.gradient(Psi, dy, dx)

Psi_star = np.conjugate(Psi)
phase_grad_y = np.imag(Psi_star * dPsi_dy)
phase_grad_x = np.imag(Psi_star * dPsi_dx)

denom = 1 + p_0 * I

# --- 4. Corrected Force Equations ---

# Dipole (Gradient) Force [Dominates due to large delta_omega]
F_grad_coeff = - (hbar * delta_omega / 2) * p_0
F_grad_x = F_grad_coeff * dI_dx / denom
F_grad_y = F_grad_coeff * dI_dy / denom

# Radiation (Scattering) Force
F_rad_coeff = (hbar * Gamma / 2) * p_0
F_rad_x = F_rad_coeff * phase_grad_x / denom
F_rad_y = F_rad_coeff * phase_grad_y / denom

# Total Force & Acceleration
F_total_x = F_grad_x + F_rad_x
F_total_y = F_grad_y + F_rad_y

a_x = F_total_x / m_Rb87*.17  # *.1 for debugging
a_y = F_total_y / m_Rb87*.17
a_mag = np.sqrt(a_x**2 + a_y**2)

print(f"Max Gradient Acceleration: {
      np.max(np.sqrt(F_grad_x**2 + F_grad_y**2)/m_Rb87):.2e} m/s^2")
print(f"Max Radiation Acceleration: {
      np.max(np.sqrt(F_rad_x**2 + F_rad_y**2)/m_Rb87):.2e} m/s^2")
print(f"Max Total Acceleration: {np.max(a_mag):.2e} m/s^2")

# %%
# --- 5. Visualization ---
# fig, ax = plt.subplots(figsize=(10, 8))
ny, nx = a_x.shape

# Center the spatial coordinates
x = (np.arange(nx) - nx/2) * dx * 1e6
y = (np.arange(ny) - ny/2) * dy * 1e6
X, Y = np.meshgrid(x, y)

# heatmap = ax.pcolormesh(X, Y, a_mag, shading='auto', cmap='inferno')
# cbar = fig.colorbar(heatmap, ax=ax)
# cbar.set_label('Acceleration Magnitude ($m/s^2$)', rotation=270, labelpad=15)

# # Plot streamlines (flow lines of the force)
# ax.streamplot(X, Y, a_x, a_y, color='white',
#               linewidth=1, density=1.5, arrowsize=1.5)

# ax.set_title("Weber Beam Acceleration Field (Corrected)")
# ax.set_xlabel("Horizontal Position ($\mu m$)")
# ax.set_ylabel("Vertical Position ($\mu m$)")
# ax.invert_yaxis()

# plt.tight_layout()
# # plt.savefig('weber_corrected_acceleration.png')
# plt.show()

# %%
plt.quiver(X, Y, a_x, a_y, I, width=0.001)
# %%


def generate_initial_cloud(N_atoms=120000, T=300e-9):
    """Generates the initial free-falling Rb87 cloud at x = -140 um."""
    m_Rb87 = 1.443160648e-25
    kB = constants.k

    def get_sigma(omega):
        return np.sqrt((kB * T) / (m_Rb87 * (2*np.pi*omega)**2))

    sig_v = np.sqrt((kB * T) / m_Rb87)

    n_half = int(N_atoms / 2)

    # Spatial distributions
    sig_x1, sig_y1 = get_sigma(247.0), get_sigma(45.0)
    sig_x2, sig_y2 = get_sigma(174.5), get_sigma(31.8)

    x0, y0 = -140e-6, 0.0

    x = np.concatenate([np.random.normal(x0, sig_x1, n_half),
                       np.random.normal(x0, sig_x2, n_half)])
    y = np.concatenate([np.random.normal(y0, sig_y1, n_half),
                       np.random.normal(y0, sig_y2, n_half)])
    vx = np.random.normal(0, sig_v, N_atoms)
    vy = np.random.normal(0, sig_v, N_atoms)

    return x, y, vx, vy


# %%
dx = dy = 0.8e-6
ny, nx = Psi.shape

# Create coordinate arrays for the grid
# x is vertical (rows of matrix mapped to x), y is horizontal
grid_x_coords = (np.arange(nx) - nx/2) * dx
grid_y_coords = (np.arange(ny) - ny/2) * dy

# Get accelerations
print("Calculating Field Accelerations...")
# a_x_grid, a_y_grid, Intensity = get_weber_acceleration_grids(Psi, dx, dy)

a_x_grid = a_x
a_y_grid = a_y
Intensity = I


# Create 2D Interpolators (bounds_error=False handles atoms flying out of the beam's bounding box)
# We fill missing values with 0 so they only experience gravity outside the beam.
interp_ax = RegularGridInterpolator(
    (grid_y_coords, grid_x_coords), a_x_grid, bounds_error=False, fill_value=0.0)
interp_ay = RegularGridInterpolator(
    (grid_y_coords, grid_x_coords), a_y_grid, bounds_error=False, fill_value=0.0)

# --- 2. Initialize Cloud ---
print("Initializing Atomic Cloud...")
# Reduced to 5k for faster testing
x, y, vx, vy = generate_initial_cloud(N_atoms=5000)
g_grav = 9.81

# --- 3. Time Integration (Euler-Cromer) ---
print("Starting Time Evolution...")

t_max = 18e-3     # Total simulation time (18 ms)
dt = 5e-6         # 5 microseconds timestep for stability inside the optical potential
steps = int(t_max / dt)

# Arrays to store snapshots for plotting
snapshots = {}
record_times = [0.0, 7e-3, 13e-3, 18e-3]

for step in range(steps):
    t_current = step * dt

    # Record snapshots at specific times
    for rt in record_times:
        if abs(t_current - rt) < dt/2:
            snapshots[rt] = (x.copy(), y.copy())
            print(f"  Recorded snapshot at t = {rt*1000:.1f} ms")

    # Combine coordinates for interpolation: shape (N, 2)
    # Note: RegularGridInterpolator expects (row_coord, col_coord) -> (y, x)
    points = np.column_stack((y, x))

    # Get optical acceleration at current positions
    ax_opt = interp_ax(points)*1  # *0 for debug
    ay_opt = interp_ay(points)

    # Update Velocities (Gravity + Optical)
    vx += (g_grav - ax_opt) * dt
    vy += ay_opt * dt

    # Update Positions
    x += vx * dt
    y += vy * dt

# Final snapshot
snapshots[t_max] = (x.copy(), y.copy())

# --- 4. Plotting Results ---
print("Generating Plots...")
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True)

# Plot intensity as background
X, Y = np.meshgrid(grid_x_coords * 1e6, grid_y_coords * 1e6)

for i, (rt, (x_snap, y_snap)) in enumerate(sorted(snapshots.items())):
    ax = axes[i]

    # Plot beam footprint
    ax.pcolormesh(X, Y, Intensity, shading='auto',
                  cmap='Greys_r', alpha=0.3)

    # Plot atom cloud histogram
    ax.hist2d(y_snap * 1e6, x_snap * 1e6, bins=100,
              range=[[-300, 300], [-300, 1800]], cmap='inferno')

    ax.set_title(f"t = {rt*1000:.1f} ms")
    ax.set_xlabel("Horizontal Position ($\mu m$)")
    if i == 0:
        ax.set_ylabel(
            "Vertical Position ($\mu m$)\n(Gravity $\downarrow$)")

    # Beam center reference
    ax.axhline(0, color='cyan', linestyle='--', alpha=0.5)

plt.suptitle("Cloud Scattering off Weber Beam", fontsize=16)
plt.gca().invert_yaxis()  # Match gravity direction
plt.tight_layout()
plt.show()
