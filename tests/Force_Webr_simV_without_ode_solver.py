# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 22:11:04 2026

@author: clher
"""

import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.interpolate import RegularGridInterpolator
import time

import pandas as pd
from scipy import constants
import scipy.io
from pathlib import Path
from datetime import datetime

# Method 1: Modern approach (Recommended)
rng = np.random.default_rng(42)
# random_numbers = rng.random(5)

# Method 2: Legacy approach (Older code)
np.random.seed(42)
# random_numbers_legacy = np.random.rand(5)


def safe_complex(x):  # A helper function to safely parse complex strings like '(5.89e-01+1.03e-02j)'
    if pd.isna(x):
        return 0j
    if isinstance(x, str):
        return complex(x.strip(' ()').replace(' ', ''))
# %%


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
    F_grad_coeff = - (hbar * delta_omega / 2) * p_0  # *10 for debugging
    F_grad_x = F_grad_coeff * dI_dx / denom
    F_grad_y = F_grad_coeff * dI_dy / denom

    F_rad_coeff = (hbar * Gamma / 2) * p_0
    F_rad_x = F_rad_coeff * phase_grad_x / denom
    F_rad_y = F_rad_coeff * phase_grad_y / denom

    # Accelerations
    a_x = (F_grad_x + F_rad_x) / m_Rb87  # *.2 for debugging
    a_y = (F_grad_y + F_rad_y) / m_Rb87 - 9.8  # gravity added

    return a_x, a_y, I


def generate_initial_cloud(N_atoms=1200, T=300e-9):
    """Generates the initial free-falling Rb87 cloud at y = -140 um."""
    m_Rb87 = 1.443160648e-25
    kB = constants.k

    def get_sigma(omega):
        return np.sqrt((kB * T) / (m_Rb87 * (2*np.pi*omega)**2))

    sig_v = np.sqrt((kB * T) / m_Rb87)

    n_half = int(N_atoms / 2)

    # Spatial distributions
    sig_y1, sig_x1 = get_sigma(247.0), get_sigma(45.0)
    sig_y2, sig_x2 = get_sigma(174.5), get_sigma(31.8)

    x0 = 0.0
    y0 = 140e-6

    # x = np.concatenate([np.random.normal(x0, sig_x1, n_half),
    #                    np.random.normal(x0, sig_x2, n_half)])
    # y = np.concatenate([np.random.normal(y0, sig_y1, n_half),
    #                    np.random.normal(y0, sig_y2, n_half)])
    # vx = np.random.normal(0, sig_v, N_atoms)
    # vy = np.random.normal(0, sig_v, N_atoms)

    x = np.concatenate([rng.normal(x0, sig_x1, n_half),
                       rng.normal(x0, sig_x2, n_half)])
    y = np.concatenate([rng.normal(y0, sig_y1, n_half),
                       rng.normal(y0, sig_y2, n_half)])
    vx = rng.normal(0, sig_v, N_atoms)
    vy = rng.normal(0, sig_v, N_atoms)

    return x, y, vx, vy

# ---------------------------------------------------------
#  Reusable CCD Camera Function
# ---------------------------------------------------------


def plot_ccd_density(t_target, x_sol, y_sol, t_eval, pixel_size=.1):
    """
    Plots a CCD-style density map of the particle cloud at a specific time.
    The axes automatically adjust tightly to the cloud's current size.
    """
    # Find the closest time index in our simulation results
    idx = np.argmin(np.abs(t_eval - t_target))
    actual_t = t_eval[idx]

    x, y = x_sol[:, idx], y_sol[:, idx]

    # Auto-adjusting limits based on the cloud's exact bounds
    x_min, x_max = np.floor(x.min()), np.ceil(x.max())
    y_min, y_max = np.floor(y.min()), np.ceil(y.max())

    # Create square bins representing CCD pixels
    x_bins = np.arange(x_min, x_max + pixel_size, pixel_size)
    y_bins = np.arange(y_min, y_max + pixel_size, pixel_size)

    fig, ax = plt.subplots(figsize=(7, 6))

    # hist2d creates the square CCD pixels. cmin=1 leaves empty pixels blank.
    counts, _, _, im = ax.hist2d(
        x, y, bins=[x_bins, y_bins], cmap='plasma', cmin=1)

    # Draw the boundary of the force field if it falls in the camera view
    rect_ref = Rectangle((0, 0), 10, 10, linewidth=1.5,
                         edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect_ref)

    fig.colorbar(im, ax=ax, label='Particles per Pixel')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')  # Forces pixels to be perfectly square

    ax.set_title(f'Auto-Scaled CCD Camera View (t = {actual_t:.2f}s)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    plt.show()


def particle_cloud_motion(t, state):
    x, y = state[0:Np], state[Np:2*Np]
    vx, vy = state[2*Np:3*Np], state[3*Np:4*Np]
    points = np.column_stack((y, x))
    Fx = interp_fx(points)
    Fy = interp_fy(points)
    # Fx = ffx(points)
    # return np.concatenate((vx, vy, interp_fx(points)/m - g*0, (interp_fy(points)/m) - g)) #changed direction of g
    return np.concatenate((vx, vy, Fx, (Fy)))

# %%
# ---------------------------------------------------------
# 1. Simulation Setup
# ---------------------------------------------------------
# Np = 500
# --- 1. Load and Parse the Psi Matrix ---


dir0 = Path(r'C:\Users\clher\Documents\programacion\projects\ultracold_Rb87')
folder_path = Path(
    r"C:\Users\clher\Documents\programacion\projects\ultracold_Rb87\data")

Psi_path = dir0 / 'Psi.npy'
Psi = np.load(Psi_path)
# Psi = np.load('Psi_cro_324.npy')
# Psi = np.load('/content/Psi_cropped_small.npy')


# Psi_path = folder_path / 'Psi_a10_kp3.mat'

# mat_data = scipy.io.loadmat(Psi_path)  # load matlab mat file

# Psi_or = mat_data['Psi']  # psi original

# df = pd.DataFrame(Psi_or)
# %%
# Psi = df.map(safe_complex).values if hasattr(
#     df, 'map') else df.applymap(safe_complex).values

# Normalize Psi so max amplitude is 1
# Psi_or = Psi / np.max(np.abs(Psi))  # psi original, to be cropped (or not)

# %% Crop Psi
# m, n = Psi_or.shape
# crf = .35  # .477  # crop factor #to crop psi original
# mc = round(m*crf)
# nc = round(n*crf)
# Psi = Psi_or[mc:-mc, nc:-nc]  # psi cropped
print(Psi.shape)

# %%

dx = dy = 0.8e-6  # pixel size times telescope reduction
ny, nx = Psi.shape

x_sys_coord = (np.arange(nx) - nx/2) * dx
y_sys_coord = (np.arange(ny) - ny/2) * dy

# add indexing='ji' if ny, nx, these are taken from coords instead columns and rows
X_grid, Y_grid = np.meshgrid(x_sys_coord, y_sys_coord)

# %%
# Get accelerations
print("Calculating Field Accelerations...")
ffx, ffy, Intensity = get_weber_acceleration_grids(
    Psi, dx, dy)  # ffx,ffy are already accelerations, not force
print("accelerations, complete")
print(ffx.shape, ffy.shape, Intensity.shape)
# #%%
# plt.figure()
# plt.quiver(X_grid, Y_grid, ffx, ffy, Intensity)
# plt.show()

# %%
# @title Set the atom cloud initial conditions
Np = 50
# --- 2. Initialize Cloud ---
print("Initializing Atomic Cloud...")
x, y, vx, vy = generate_initial_cloud(
    N_atoms=Np)  # Reduced to 1k for faster testing

# Pack initial conditions into a single 1D array
state0 = np.concatenate((x, y, vx, vy))
# %%
interp_fx = RegularGridInterpolator(
    (y_sys_coord, x_sys_coord), ffx, method='cubic', bounds_error=False, fill_value=0.0)
interp_fy = RegularGridInterpolator(
    (y_sys_coord, x_sys_coord), ffy, method='cubic', bounds_error=False, fill_value=-9.8)
# %%

interp_ax = interp_fx
interp_ay = interp_fy
# --- 3. Time Integration (Euler-Cromer) ---
print("Starting Time Evolution...")

t_max = 18e-3     # Total simulation time (18 ms)
dt = 5e-6         # 5 microseconds timestep for stability inside the optical potential
steps = int(t_max / dt)
t_traj = np.zeros(int(steps/10))
x_traj = np.zeros((Np, int(steps/10)))
y_traj = np.zeros((Np, int(steps/10)))
# Arrays to store snapshots for plotting
snapshots = {}
record_times = [0.0, 7e-3, 10e-3, 12e-3]
aux = 0
start_time = time.perf_counter()
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
    ax_opt = interp_ax(points)  # *0 for debug
    ay_opt = interp_ay(points)

    # Update Velocities (Gravity already included in Optical)
    vx += (ax_opt) * dt
    vy += ay_opt * dt

    # Update Positions
    x += vx * dt
    y += vy * dt
    # print(step)
    if step % 10 == 0 and step/10 < len(t_traj):
        # print(aux)
        t_traj[aux] = t_current
        x_traj[:, aux] = x.copy()
        y_traj[:, aux] = y.copy()
        aux = aux + 1
# "toc"
end_time = time.perf_counter()
exec_time_without_ode = end_time - start_time
# %% Plot without ode solver
um_factor = 1e6

plt.plot(x_traj[5:10, :].T*um_factor, y_traj[5:10, :].T*um_factor,
         'o', mec='0.1',  ls='')

# %% to compare with ode solver

# %% LSODA
t_end = t_max  # time in seconds
t_span = (0, t_end)
t_eval = np.linspace(t_span[0], t_span[1], int(len(t_traj)/3))
print("Solving ODEs...")
# solution = solve_ivp(particle_cloud_motion, (0, t_end), state0, t_eval=t_eval, method='RK45')
print(f"Solving ODEs for {Np} particles over {t_end} seconds...")
# Radau RK45 BDF LSODA
ode_method_list = ['Radau', 'RK45', 'BDF', 'LSODA']
ode_method = ode_method_list[3]

# "tic"
start_time = time.perf_counter()

solution = solve_ivp(particle_cloud_motion, t_span,
                     state0, t_eval=t_eval, method=ode_method)

x_sol, y_sol = solution.y[0:Np, :], solution.y[Np:2*Np, :]

# "toc"
end_time = time.perf_counter()
exec_time = end_time - start_time

print(end_time-start_time)

#   change units from m to um for display
# ---------------------------------------------------------
# 3. Main 1x5 Overview Plot
# ---------------------------------------------------------
um_factor = 1e6
x_sc_mu = x_sys_coord*um_factor
y_sc_mu = x_sys_coord*um_factor

x_sol_mu = x_sol*1e6
y_sol_mu = y_sol*1e6
X_mu = X_grid * 1e6
Y_mu = Y_grid*1e6

# %%
# x_lims = (-200, 200)
# y_lims = (-200, 150)
figq, ax_q = plt.subplots(figsize=(9, 9))
# ax_q.quiver(X_mu[::2, ::2], Y]_mu[::2, ::2], ffx[::2, ::2],
#             ffy[::2, ::2], color='gray', alpha=0.4)

ax_q.pcolormesh(X_mu[::2, ::2], Y_mu[::2, ::2], abs(Psi[::2, ::2]), alpha=.3)
ax_q.plot(x_sol_mu[5:10, :].T, y_sol_mu[5:10, :].T, '.', mec='0.1',  ls='')
ax_q.set_title('Trajectories (20 Particles)')
ax_q.set_title(f'exec_time={exec_time:.2f}s \t (ode = {
               ode_method}) \n exec_time_without_ode={exec_time_without_ode}')
# ax_q.set_xlim(x_lims)
# ax_q.set_ylim(y_lims)
plt.show()

# %%

t_end = t_max  # time in seconds
t_span = (0, t_end)
t_eval = np.linspace(t_span[0], t_span[1], int(len(t_traj)/3))
print("Solving ODEs...")
# solution = solve_ivp(particle_cloud_motion, (0, t_end), state0, t_eval=t_eval, method='RK45')
print(f"Solving ODEs for {Np} particles over {t_end} seconds...")
# Radau RK45 BDF LSODA
ode_method_list = ['Radau', 'RK45', 'BDF', 'LSODA']
ode_method = ode_method_list[0]

# "tic"
start_time = time.perf_counter()

solution = solve_ivp(particle_cloud_motion, t_span,
                     state0, t_eval=t_eval, method=ode_method)

x_sol, y_sol = solution.y[0:Np, :], solution.y[Np:2*Np, :]

# "toc"
end_time = time.perf_counter()
exec_time = end_time - start_time

print(end_time-start_time)

#   change units from m to um for display
# ---------------------------------------------------------
# 3. Main 1x5 Overview Plot
# ---------------------------------------------------------
um_factor = 1e6
x_sc_mu = x_sys_coord*um_factor
y_sc_mu = x_sys_coord*um_factor

x_sol_mu = x_sol*1e6
y_sol_mu = y_sol*1e6
X_mu = X_grid * 1e6
Y_mu = Y_grid*1e6

# %%
# x_lims = (-200, 200)
# y_lims = (-200, 150)
figq, ax_q = plt.subplots(figsize=(9, 9))
# ax_q.quiver(X_mu[::2, ::2], Y]_mu[::2, ::2], ffx[::2, ::2],
#             ffy[::2, ::2], color='gray', alpha=0.4)

ax_q.pcolormesh(X_mu[::2, ::2], Y_mu[::2, ::2], abs(Psi[::2, ::2]), alpha=.3)
ax_q.plot(x_sol_mu[5:10, :].T, y_sol_mu[5:10, :].T, '.', mec='0.1',  ls='')
ax_q.set_title('Trajectories (20 Particles)')
ax_q.set_title(f'exec_time={exec_time:.2f}s \t (ode = {
               ode_method}) \n exec_time_without_ode={exec_time_without_ode}')
# ax_q.set_xlim(x_lims)
# ax_q.set_ylim(y_lims)
plt.show()

# %% RK45
t_end = t_max  # time in seconds
t_span = (0, t_end)
t_eval = np.linspace(t_span[0], t_span[1], int(len(t_traj)/3))
print("Solving ODEs...")
# solution = solve_ivp(particle_cloud_motion, (0, t_end), state0, t_eval=t_eval, method='RK45')
print(f"Solving ODEs for {Np} particles over {t_end} seconds...")
# Radau RK45 BDF LSODA
ode_method_list = ['Radau', 'RK45', 'BDF', 'LSODA']
ode_method = ode_method_list[1]

# "tic"
start_time = time.perf_counter()

solution = solve_ivp(particle_cloud_motion, t_span,
                     state0, t_eval=t_eval, method=ode_method)

x_sol, y_sol = solution.y[0:Np, :], solution.y[Np:2*Np, :]

# "toc"
end_time = time.perf_counter()
exec_time = end_time - start_time

print(end_time-start_time)

#   change units from m to um for display
# ---------------------------------------------------------
# 3. Main 1x5 Overview Plot
# ---------------------------------------------------------
um_factor = 1e6
x_sc_mu = x_sys_coord*um_factor
y_sc_mu = x_sys_coord*um_factor

x_sol_mu = x_sol*1e6
y_sol_mu = y_sol*1e6
X_mu = X_grid * 1e6
Y_mu = Y_grid*1e6


# x_lims = (-200, 200)
# y_lims = (-200, 150)
figq, ax_q = plt.subplots(figsize=(9, 9))
# ax_q.quiver(X_mu[::2, ::2], Y]_mu[::2, ::2], ffx[::2, ::2],
#             ffy[::2, ::2], color='gray', alpha=0.4)

ax_q.pcolormesh(X_mu[::2, ::2], Y_mu[::2, ::2], abs(Psi[::2, ::2]), alpha=.3)
ax_q.plot(x_sol_mu[5:10, :].T, y_sol_mu[5:10, :].T, '.', mec='0.1',  ls='')
ax_q.set_title('Trajectories (20 Particles)')
ax_q.set_title(f'exec_time={exec_time:.2f}s \t (ode = {
               ode_method}) \n exec_time_without_ode={exec_time_without_ode}')
# ax_q.set_xlim(x_lims)
# ax_q.set_ylim(y_lims)
plt.show()

# %% RK23
print("Solving ODEs...")
# solution = solve_ivp(particle_cloud_motion, (0, t_end), state0, t_eval=t_eval, method='RK45')
print(f"Solving ODEs for {Np} particles over {t_end} seconds...")
# Radau RK45 BDF LSODA
ode_method_list = ['Radau', 'RK45', 'BDF', 'LSODA', 'RK23']
ode_method = ode_method_list[4]

# "tic"
start_time = time.perf_counter()

solution = solve_ivp(particle_cloud_motion, t_span,
                     state0, t_eval=t_eval, method=ode_method)

x_sol, y_sol = solution.y[0:Np, :], solution.y[Np:2*Np, :]

# "toc"
end_time = time.perf_counter()
exec_time = end_time - start_time

print(end_time-start_time)

#   change units from m to um for display
# ---------------------------------------------------------
# 3. Main 1x5 Overview Plot
# ---------------------------------------------------------
um_factor = 1e6
x_sc_mu = x_sys_coord*um_factor
y_sc_mu = x_sys_coord*um_factor

x_sol_mu = x_sol*1e6
y_sol_mu = y_sol*1e6
X_mu = X_grid * 1e6
Y_mu = Y_grid*1e6


# x_lims = (-200, 200)
# y_lims = (-200, 150)
figq, ax_q = plt.subplots(figsize=(9, 9))
# ax_q.quiver(X_mu[::2, ::2], Y]_mu[::2, ::2], ffx[::2, ::2],
#             ffy[::2, ::2], color='gray', alpha=0.4)

ax_q.pcolormesh(X_mu[::2, ::2], Y_mu[::2, ::2], abs(Psi[::2, ::2]), alpha=.3)
ax_q.plot(x_sol_mu[5:10, :].T, y_sol_mu[5:10, :].T, '.', mec='0.1',  ls='')
ax_q.set_title('Trajectories (20 Particles)')
ax_q.set_title(f'exec_time={exec_time:.2f}s \t (ode = {
               ode_method}) \n exec_time_without_ode={exec_time_without_ode}')
# ax_q.set_xlim(x_lims)
# ax_q.set_ylim(y_lims)
plt.show()
