# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 22:07:51 2026

@author: clher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import scipy.io
from pathlib import Path
from datetime import datetime


# --- 1. Load and Parse the Psi Matrix ---
# A helper function to safely parse complex strings like '(5.89e-01+1.03e-02j)'
def safe_complex(x):
    if pd.isna(x):
        return 0j
    if isinstance(x, str):
        return complex(x.strip(' ()').replace(' ', ''))
    return complex(x)


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

a_x = F_total_x / m_Rb87
a_y = F_total_y / m_Rb87
a_mag = np.sqrt(a_x**2 + a_y**2)

print(f"Max Gradient Acceleration: {
      np.max(np.sqrt(F_grad_x**2 + F_grad_y**2)/m_Rb87):.2e} m/s^2")
print(f"Max Radiation Acceleration: {
      np.max(np.sqrt(F_rad_x**2 + F_rad_y**2)/m_Rb87):.2e} m/s^2")
print(f"Max Total Acceleration: {np.max(a_mag):.2e} m/s^2")

# %%
# --- 5. Visualization ---
fig, ax = plt.subplots(figsize=(10, 8))
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
plt.quiver(X, Y, a_x, a_y, width=0.001)
