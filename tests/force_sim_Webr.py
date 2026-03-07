# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 19:13:02 2026

@author: clher
"""

"""
Calculates the acceleration field for Rb87 atoms given a complex Weber field.

Psi: 2D complex numpy array, normalized so max(abs(Psi)) == 1
dx, dy: Spatial resolution of the matrix in meters (e.g., 1e-6 for 1 um)
g_factor: Ratio of max Rabi frequency to Gamma (e.g., 19.1 from the paper)
"""


# Add src to path - MUST be before importing ultracold modules
import plotly.io as pio
import plotly.figure_factory as ff
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import scipy.io
from pathlib import Path
from datetime import datetime
PROJECT_ROOT = Path(__file__).parent.parent

pio.renderers.default = 'browser'
# Detect environment
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None and 'IPKernelApp' in ip.config:
        # We are in a Jupyter notebook
        pio.renderers.default = 'notebook'
    else:
        # Probably Spyder or other console – use browser
        pio.renderers.default = 'browser'
except:
    # Fallback – no IPython, use browser
    pio.renderers.default = 'browser'

pio.renderers.default = 'browser'


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


dir_load = r'C:\Users\clher\Documents\programacion\projects\ultracold_Rb87\scripts'
folder_path = Path(
    r"C:\Users\clher\Documents\programacion\projects\ultracold_Rb87\data")
full_path = folder_path / 'Psi_a10_kp3.mat'

# dirs = setup_output_directories(output_base)
# print(f"Output directory: {dirs['data']}")

# --- Example Usage (Assuming you have a_x, a_y, dx, dy ready) ---
mat_data = scipy.io.loadmat(full_path)

Psi_or = mat_data['Psi']  # psi original
# %% Crop Psi
m, n = Psi_or.shape
crf = .477  # crop factor #to crop psi original
mc = round(m*crf)
nc = round(n*crf)
Psi = Psi_or[mc-30:-mc-30, nc:-nc]  # psi cropped
print(Psi.shape)
# %%
dx = .8e-6  # e.g., 1 micron resolution
dy = .8e-6

g_factor = 19.1

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
F_grad_coeff = hbar * delta_omega * p_0 / denom
F_grad_x = F_grad_coeff * dI_dx
F_grad_y = F_grad_coeff * dI_dy

# 2. Radiation Pressure Force (Phase force)
F_rad_coeff = (hbar * Gamma / 4) * p_0 / denom
F_rad_x = F_rad_coeff * phase_grad_x
F_rad_y = F_rad_coeff * phase_grad_y

# Total Force
F_total_x = F_grad_x + F_rad_x  # *0 for debugging
F_total_y = F_grad_y + F_rad_y  # *0 for debugging

# Convert to Acceleration
a_x = F_total_x / m_Rb87
a_y = F_total_y / m_Rb87

# %%
# 2. Create the spatial grids for the axes
# Assuming the center of the matrix is (0,0)
ny, nx = a_x.shape
x = (np.arange(nx) - nx/2) * dx * 1e6  # Convert to microns for plotting
y = (np.arange(ny) - ny/2) * dy * 1e6
X, Y = np.meshgrid(x, y)

# fig = ff.create_quiver(X, Y, a_x, a_y)
# fig.show()        # now opens in a browser tab
# %%

# %%
plt.quiver(a_x, a_y, width=0.003)
# %%
plt.quiver(X, Y, a_x, a_y, width=0.001)
# %%
# plt.quiver(X, Y, a_x, a_y, scale=2)
# %%
# plt.quiver(dI_dx, dI_dy, width=0.003)
