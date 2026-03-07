# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 22:38:32 2026

@author: clher
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


def get_weber_acceleration_grids(Psi, dx, dy, g_factor=19.1):
    """Calculates the acceleration grids a_x and a_y from the Psi matrix."""
    # Normalize
    Psi = Psi / np.max(np.abs(Psi))

    # Constants
    hbar = constants.hbar
    c = constants.c
    m_Rb87 = 1.443160648e-25
    lambda_0 = 780.241e-9
    lambda_L = 780.368e-9
    Gamma = 2 * np.pi * 6.06e6

    # Frequencies
    omega_0 = 2 * np.pi * c / lambda_0
    omega_L = 2 * np.pi * c / lambda_L
    delta_omega = omega_L - omega_0

    g = g_factor * Gamma
    p_0 = (2 * g**2) / ((Gamma / 2)**2 + delta_omega**2)

    # Field gradients
    I = np.abs(Psi)**2
    dI_dy, dI_dx = np.gradient(I, dy, dx)

    dPsi_dy, dPsi_dx = np.gradient(Psi, dy, dx)
    Psi_star = np.conjugate(Psi)
    phase_grad_y = np.imag(Psi_star * dPsi_dy)
    phase_grad_x = np.imag(Psi_star * dPsi_dx)

    denom = 1 + p_0 * I

    # Forces
    F_grad_coeff = - (hbar * delta_omega / 2) * p_0
    F_grad_x = F_grad_coeff * dI_dx / denom
    F_grad_y = F_grad_coeff * dI_dy / denom

    F_rad_coeff = (hbar * Gamma / 2) * p_0
    F_rad_x = F_rad_coeff * phase_grad_x / denom
    F_rad_y = F_rad_coeff * phase_grad_y / denom

    # Accelerations
    a_x = (F_grad_x + F_rad_x) / m_Rb87
    a_y = (F_grad_y + F_rad_y) / m_Rb87

    return a_x, a_y, I


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

    x0 = 0.0
    y0 = -140e-6

    x = np.concatenate([np.random.normal(x0, sig_x1, n_half),
                       np.random.normal(x0, sig_x2, n_half)])
    y = np.concatenate([np.random.normal(y0, sig_y1, n_half),
                       np.random.normal(y0, sig_y2, n_half)])
    vx = np.random.normal(0, sig_v, N_atoms)
    vy = np.random.normal(0, sig_v, N_atoms)

    return x, y, vx, vy


def run_scattering_simulation():
    # --- 1. Load Your Psi Matrix Here ---
    path_psi_cr = Path(
        r'C:/Users/clher/Documents/programacion/projects/ultracold_Rb87/tests/Psi_cropped.csv')
    # Replace this with loading your full Psi matrix
    print("Loading Field...")
    try:
        # Example: if you have a local Psi.npy or Psi.csv
        # import pandas as pd
        # df = pd.read_csv(path_psi_cr, header=None)

        # def safe_complex(x):
        #     if pd.isna(x):
        #         return 0j
        #     if isinstance(x, str):
        #         return complex(x.strip(' ()').replace(' ', ''))
        #     return complex(x)
        # Psi = df.map(safe_complex).values

        # # FOR NOW: Creating a dummy 2D Gaussian beam so the script runs out of the box
        # grid_y, grid_x = np.mgrid[-200:200:400j, -200:200:400j]
        # Psi = np.exp(-(grid_x**2 + grid_y**2)/(2*50**2)) * \
        #     np.exp(1j * 0.1 * grid_x)
        # Psi = np.load(path_psi_cr)
        # Psi = np.loadtxt(path_psi_cr, delimiter=',')
        Psi = np.load('Psi_cro_324.npy')
        # Psi = np.load('Psi_cropped_small.npy')
        Psi = np.flipud(Psi)

    except Exception as e:
        print(f"Error loading matrix: {e}")
        return

    dx = dy = 0.8e-6
    ny, nx = Psi.shape

    # Create coordinate arrays for the grid
    # x is vertical (rows of matrix mapped to x), y is horizontal
    grid_x_coords = (np.arange(nx) - nx/2) * dx
    grid_y_coords = (np.arange(ny) - ny/2) * dy

    # Get accelerations
    print("Calculating Field Accelerations...")
    a_x_grid, a_y_grid, Intensity = get_weber_acceleration_grids(Psi, dx, dy)

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
        ax_opt = interp_ax(points)
        ay_opt = interp_ay(points)

        # Update Velocities (Gravity + Optical)
        vx += (g_grav + ax_opt) * dt
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


if __name__ == "__main__":
    run_scattering_simulation()
