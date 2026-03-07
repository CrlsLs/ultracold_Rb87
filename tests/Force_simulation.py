# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 22:29:43 2026

@author: clher
"""
import plotly.figure_factory as ff
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import scipy.io
from pathlib import Path
from datetime import datetime


# Add src to path - MUST be before importing ultracold modules
PROJECT_ROOT = Path(__file__).parent.parent


def setup_output_directories(output_base: Path) -> dict:
    """Create output directories and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dirs = {
        'base': output_base,
        'figures': output_base / 'figures' / f'Webr_accel_field_{timestamp}',
        'data': output_base / 'simulations' / f'Webr_accel_field_{timestamp}',
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def calculate_weber_acceleration(Psi, dx, dy, g_factor=19.1):
    """
    Calculates the acceleration field for Rb87 atoms given a complex Weber field.

    Psi: 2D complex numpy array, normalized so max(abs(Psi)) == 1
    dx, dy: Spatial resolution of the matrix in meters (e.g., 1e-6 for 1 um)
    g_factor: Ratio of max Rabi frequency to Gamma (e.g., 19.1 from the paper)
    """
    # --- Physics Constants ---
    hbar = constants.hbar
    c = constants.c
    m_Rb87 = 1.443160648e-25  # kg

    # --- Paper Parameters ---
    lambda_0 = 780.368e-9     # m
    detuning_nm = 0.127e-9    # m (red detuned)
    Gamma = 2 * np.pi * 6.06e6  # rad/s

    # --- Derived Angular Frequencies ---
    omega_0 = 2 * np.pi * c / lambda_0
    omega_laser = 2 * np.pi * c / (lambda_0 + detuning_nm)
    delta_omega = omega_laser - omega_0  # Expected ~ -3.93e11 rad/s

    g = g_factor * Gamma
    p_0 = (2 * g**2) / ((Gamma / 2)**2 + delta_omega**2)

    # --- Field Calculations ---
    I = np.abs(Psi)**2

    # Gradients of Intensity
    # Assuming Psi uses (row=y, col=x) standard
    dI_dy, dI_dx = np.gradient(I, dy, dx)

    # Gradients of the Complex Field
    dPsi_dy, dPsi_dx = np.gradient(Psi, dy, dx)

    # Phase Gradients using Im(Psi* \nabla Psi)
    Psi_star = np.conjugate(Psi)
    phase_grad_y = np.imag(Psi_star * dPsi_dy)
    phase_grad_x = np.imag(Psi_star * dPsi_dx)

    # Common denominator
    denom = 1 + p_0 * I

    # --- Force Calculations ---
    # 1. Gradient Force (Dipole force)
    F_grad_coeff = (hbar * Gamma / 4) * p_0 / denom
    F_grad_x = F_grad_coeff * dI_dx
    F_grad_y = F_grad_coeff * dI_dy

    # 2. Radiation Pressure Force (Phase force)
    F_rad_coeff = hbar * delta_omega * p_0 / denom
    F_rad_x = F_rad_coeff * phase_grad_x
    F_rad_y = F_rad_coeff * phase_grad_y

    # Total Force
    F_total_x = F_grad_x + F_rad_x*0  # *0 for debugging
    F_total_y = F_grad_y + F_rad_y*0  # *0 for debugging

    # Convert to Acceleration
    a_x = F_total_x / m_Rb87
    a_y = F_total_y / m_Rb87

    return a_x, a_y


def plot_acceleration_field(a_x, a_y, dx, dy, downsample=1):
    """
    Visualizes the acceleration field using a streamplot and magnitude heatmap.

    a_x, a_y: 2D numpy arrays of the acceleration components.
    dx, dy: Spatial resolution in meters.
    downsample: Integer to reduce arrow clutter if using quiver (not strictly needed for streamplot, but good for large matrices).
    """
    # 1. Calculate the magnitude of the acceleration
    a_mag = np.sqrt(a_x**2 + a_y**2)

    # 2. Create the spatial grids for the axes
    # Assuming the center of the matrix is (0,0)
    ny, nx = a_x.shape
    x = (np.arange(nx) - nx/2) * dx * 1e6  # Convert to microns for plotting
    y = (np.arange(ny) - ny/2) * dy * 1e6
    X, Y = np.meshgrid(x, y)

    # 3. Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the magnitude as a background heatmap
    # Using log scale can help if the intensity varies massively
    heatmap = ax.pcolormesh(X, Y, a_mag, shading='auto', cmap='viridis')
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Acceleration Magnitude ($m/s^2$)',
                   rotation=270, labelpad=15)

    # 4. Plot the streamlines
    # Streamplot handles dense grids very well
    # We use the raw a_x and a_y here. Note: check if a_x aligns with X and a_y with Y based on your meshgrid setup!
    ax.streamplot(X, Y, a_x, a_y, color='white',
                  linewidth=1, density=1.5, arrowsize=1.5)

    # Alternative: Quiver plot (uncomment to use instead of streamplot)
    # ds = downsample
    # ax.quiver(X[::ds, ::ds], Y[::ds, ::ds], a_x[::ds, ::ds], a_y[::ds, ::ds],
    #           color='white', pivot='mid')

    # 5. Formatting
    ax.set_title("Weber Beam Acceleration Field")
    ax.set_xlabel("Horizontal Position ($\mu m$)")
    ax.set_ylabel("Vertical Position ($\mu m$)")

    # Invert Y axis if you want to match the "gravity pointing down" convention from earlier
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# %%
parser = argparse.ArgumentParser(
    description='Simulate Weber forces on ultracold Rb-87 atomic cloud',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--output-dir', type=str, default='output/simulations',
    help='Output directory (default: output/simulations)'
)

args = parser.parse_args()

output_base = Path(PROJECT_ROOT / 'output')

dir_load = r'C:\Users\clher\Documents\programacion\projects\ultracold_Rb87\scripts'
folder_path = Path(
    r"C:\Users\clher\Documents\programacion\projects\ultracold_Rb87\data")
full_path = folder_path / 'Psi_a10_kp3.mat'

dirs = setup_output_directories(output_base)
print(f"Output directory: {dirs['data']}")

# --- Example Usage (Assuming you have a_x, a_y, dx, dy ready) ---
mat_data = scipy.io.loadmat(full_path)

Psi = mat_data['Psi']
m, n = Psi.shape
mc = round(m/5*2)
nc = round(n/5*2)
psic = Psi[mc:-mc, nc:-nc]


dx = .8e-6  # e.g., 1 micron resolution
dy = .8e-6
# %%
accw_x, accw_y = calculate_weber_acceleration(psic, dx, dy, g_factor=19.1)
# %%
plot_acceleration_field(accw_x, accw_y, dx, dy)
# %%
plt.quiver(accw_x, accw_y)

# %%
# --- Field Calculations ---
I = np.abs(psic)**2

# Gradients of Intensity
# Assuming Psi uses (row=y, col=x) standard
dI_dy, dI_dx = np.gradient(I, dy, dx)

# Gradients of the Complex Field
dPsi_dy, dPsi_dx = np.gradient(Psi, dy, dx)
# %%

# %%

"""
Visualizes the acceleration field using a streamplot and magnitude heatmap.

a_x, a_y: 2D numpy arrays of the acceleration components.
dx, dy: Spatial resolution in meters.
downsample: Integer to reduce arrow clutter if using quiver (not strictly needed for streamplot, but good for large matrices).
"""
a_x = dI_dy
a_y = dI_dx

# 1. Calculate the magnitude of the acceleration
a_mag = np.sqrt(a_x**2 + a_y**2)

# 2. Create the spatial grids for the axes
# Assuming the center of the matrix is (0,0)
ny, nx = a_x.shape
x = (np.arange(nx) - nx/2) * dx * 1e6  # Convert to microns for plotting
y = (np.arange(ny) - ny/2) * dy * 1e6
X, Y = np.meshgrid(x, y)

# 3. Set up the plot
# fig, ax = plt.subplots(figsize=(10, 8))

# Plot the magnitude as a background heatmap
# Using log scale can help if the intensity varies massively
# heatmap = ax.pcolormesh(X, Y, a_mag, shading='auto', cmap='viridis')
# cbar = fig.colorbar(heatmap, ax=ax)
# cbar.set_label('Acceleration Magnitude ($m/s^2$)',
#                rotation=270, labelpad=15)

# 4. Plot the streamlines
# Streamplot handles dense grids very well
# We use the raw a_x and a_y here. Note: check if a_x aligns with X and a_y with Y based on your meshgrid setup!
# # ax.streamplot(X, Y, a_x, a_y, color='blue',
#               linewidth=1, density=2, arrowsize=1)

# # Alternative: Quiver plot (uncomment to use instead of streamplot)
# # ds = downsample
# # ax.quiver(X[::ds, ::ds], Y[::ds, ::ds], a_x[::ds, ::ds], a_y[::ds, ::ds],
# #           color='white', pivot='mid')

# # 5. Formatting
# ax.set_title("Weber Beam Acceleration Field")
# ax.set_xlabel("Horizontal Position ($\mu m$)")
# ax.set_ylabel("Vertical Position ($\mu m$)")

# # Invert Y axis if you want to match the "gravity pointing down" convention from earlier
# ax.invert_yaxis()

# plt.tight_layout()
# plt.show()
# %%
U = a_x
V = a_y
fig1, ax1 = plt.subplots()
ax1.set_title('Arrows scale with plot width, not view')
Q = ax1.quiver(X, Y, U, V, units='width')
qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
# %%
# Creating plot
fig, ax = plt.subplots(figsize=(9, 9))
ax.quiver(X, Y, accw_x, accw_y, alpha=1, units='width')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_aspect('equal')

# show plot
plt.show()
# %%
fig2, ax2 = plt.subplots()
ax2.set_title("pivot='mid'; every third arrow; units='inches'")
Q = ax2.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               pivot='mid', units='inches')
qk = ax2.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
ax2.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
# %%
# fig3, ax3 = plt.subplots()

# ax3.set_title("pivot='tip'; scales with x view")
# M = np.hypot(U, V)
# Q = ax3.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022,
#                scale=1 / 0.15)
# qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
#                    coordinates='figure')
# ax3.scatter(X, Y, color='0.5', s=1)

# plt.show()

# %%
plt.quiver(X, Y, accw_x, accw_y)

# %%
fig, ax = plt.subplots(figsize=(14, 8))
ax.quiver(X, Y, a_x, a_y)

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.axis([-0.3, 2.3, -0.3, 2.3])
ax.set_aspect('equal')

# show plot
plt.show()
# %%
fig = ff.create_quiver(X, Y, a_x, a_y)
fig.show()

# %%
x, y = np.meshgrid(np.arange(0, 2, .2), np.arange(0, 2, .2))
u = np.cos(x)*y
v = np.sin(x)*y

fig = ff.create_quiver(x, y, u, v)
fig.show()
