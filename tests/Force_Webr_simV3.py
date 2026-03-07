# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 23:00:02 2026

@author: clher
"""

# %% import and set
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

# %% Define functions


def safe_complex(x):  # A helper function to safely parse complex strings like '(5.89e-01+1.03e-02j)'
    if pd.isna(x):
        return 0j
    if isinstance(x, str):
        return complex(x.strip(' ()').replace(' ', ''))


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

# %% Set paths, import Psi
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

# %% Get accelerations fields
print("Calculating Field Accelerations...")
ffx, ffy, Intensity = get_weber_acceleration_grids(
    Psi, dx, dy)  # ffx,ffy are already accelerations, not force
print("accelerations, complete")
print(ffx.shape, ffy.shape, Intensity.shape)
# #%%
# plt.figure()
# plt.quiver(X_grid, Y_grid, ffx, ffy, Intensity)
# plt.show()
# %% Interp acceleration fields
interp_fx = RegularGridInterpolator(
    (y_sys_coord, x_sys_coord), ffx, method='cubic', bounds_error=False, fill_value=0.0)
interp_fy = RegularGridInterpolator(
    (y_sys_coord, x_sys_coord), ffy, method='cubic', bounds_error=False, fill_value=-9.8)

# %% this part shows if the interpolations is behaving as expected

# 2. Create the Extended Grid
# # ---------------------------------------------------------
# # Make the limits significantly wider than the original 0-5 and 0-4 range
# w_factor = 1.03  # widering factor
# x_ext = (np.arange(nx*w_factor) - nx*w_factor/2) * dx
# y_ext = (np.arange(ny*w_factor) - ny*w_factor/2) * dy
# X_ext, Y_ext = np.meshgrid(x_ext, y_ext)

# # ---------------------------------------------------------
# # 3. Evaluate Interpolators on Extended Grid
# # ---------------------------------------------------------
# # Stack points into (N, 2).
# # Note: Because your interpolator takes (y, x), we stack Y_ext then X_ext
# eval_points = np.column_stack((Y_ext.ravel(), X_ext.ravel()))

# # Evaluate and reshape back to the 2D grid shape for plotting
# ffx_ext = interp_fx(eval_points).reshape(X_ext.shape)
# ffy_ext = interp_fy(eval_points).reshape(X_ext.shape)

# # ---------------------------------------------------------
# # 4. Plot the Results
# # ---------------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 6))

# col_lim = 60
# row_lim = 100
# # Plot the vector field
# ax.quiver(X_ext[0:col_lim, 0:row_lim], Y_ext[0:col_lim, 0:row_lim], ffx_ext[0:col_lim, 0:row_lim], ffy_ext[0:col_lim, 0:row_lim],
#           color='blue', pivot='middle', width=0.001)

# # Draw a red rectangle to show the original m x n boundary
# # rect = patches.Rectangle(
# #     (x_sys_coord.min(), y_sys_coord.min()),           # (x, y) bottom-left corner
# #     x_sys_coord.max() - x_sys_coord.min(),            # Width
# #     y_sys_coord.max() - y_sys_coord.min(),            # Height
# #     linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Original Boundary'
# # )
# # ax.add_patch(rect)

# # Formatting
# ax.set_title("Force Field Interpolation (Zero Outside Boundary)")
# ax.set_xlabel("X Coordinate")
# ax.set_ylabel("Y Coordinate")
# ax.legend(loc='upper right')
# ax.grid(True, alpha=0.3)

# plt.axis('equal')  # Keeps the aspect ratio square so vectors point accurately
# plt.show()
# ---------------------------------------------------------
# %% # 1. Simulation Setup, cloud init
# ---------------------------------------------------------
Np = 50

# --- 2. Initialize Cloud ---
print("Initializing Atomic Cloud...")
x0, y0, vx0, vy0 = generate_initial_cloud(
    N_atoms=Np)  # Reduced to 1k for faster testing

# Pack initial conditions into a single 1D array
state0 = np.concatenate((x0, y0, vx0, vy0))
# %% #2 Time setup
# t_end = 9.0e-3 #time in seconds
# t_eval = np.linspace(0, t_end, 200)

t_end = 18.0e-3  # time in seconds
t_span = (0, t_end)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# %% # 3. Solving ODEs
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
x_lims = (-200, 200)
y_lims = (-200, 150)

figq, ax_q = plt.subplots(figsize=(9, 9))
# ax_q.quiver(X_mu[::2, ::2], Y]_mu[::2, ::2], ffx[::2, ::2],
#             ffy[::2, ::2], color='gray', alpha=0.4)

ax_q.pcolormesh(X_mu[::2, ::2], Y_mu[::2, ::2], abs(Psi[::2, ::2]), alpha=.3)
ax_q.plot(x_sol_mu[1:10, :].T, y_sol_mu[1:10, :].T, '.', mec='0.7',  ls='')
ax_q.set_title('Trajectories (20 Particles)')
ax_q.set_title(f'{ode_method}\n(ode = {ode_method})')
# ax_q.set_xlim(x_lims)
# ax_q.set_ylim(y_lims)
plt.show()

# %% parabolic coords
Y_grid_forParab, X_grid_forParab = np.meshgrid(
    x_sys_coord*1e6, y_sys_coord*1e6)
caux = X_grid_forParab+1j*Y_grid_forParab
uuuu = np.abs(np.real(np.sqrt(2*caux)))
vvv = np.abs(np.imag(np.sqrt(2*caux)))
uuu = uuuu.copy()
cmap = 'jet'
plt.pcolormesh(vvv, shading='flat', cmap=cmap)
plt.colorbar(cmap=cmap)

# %% parabolic coords
caux = Y_mu-1j*X_mu
vvv = np.abs(np.real(np.sqrt(2*caux)))
uuuu = np.abs(np.imag(np.sqrt(2*caux)))
uuu = uuuu.copy()
cmap = 'jet'
plt.pcolormesh(vvv, shading='flat', cmap=cmap)
plt.colorbar(cmap=cmap)

# %%
plt.pcolormesh(uuu, shading='flat', cmap=cmap)
plt.colorbar(cmap=cmap)
# %%
plt.pcolormesh(uuu[::2, ::2], vvv[::2, ::2], np.abs(Psi[::2, ::2]), alpha=.3)
plt.colorbar(cmap=cmap)
# %%
cmap = 'jet'
# plt.pcolormesh(uuuu[0:540,:], shading='flat', cmap =cmap )
# plt.colorbar(cmap = cmap)
uuu[:, 0:959] = -uuuu[:, 0:959]
plt.pcolormesh(uuu[::2, ::2], vvv[::2, ::2], np.abs(Psi[::2, ::2]), alpha=.3)
# %%
for yi in range(0, ny):
    for xi in range(0, nx):
        if X_grid[yi, xi] < 0:
            uuu[yi, xi] = -uuuu[yi, xi]

# plt.pcolormesh(uuu, shading='flat', cmap =cmap )
# plt.colorbar(cmap = cmap)


plt.pcolormesh(uuu[::2, ::2], vvv[::2, ::2], np.abs(Psi[::2, ::2]), alpha=.3)

# %% Parabolic coordd trajectories

caux = y_sol_mu-1j*x_sol_mu
v_sol_mu = np.abs(np.real(np.sqrt(2*caux)))
uu_sol_mu = np.abs(np.imag(np.sqrt(2*caux)))
u_sol_mu = uu_sol_mu.copy()
# %%
# u_sol_mu[:,0:959] = -uu_sol_mu[:,0:959]

plt.plot(v_sol_mu[0:9, :].T, u_sol_mu[0:9, :].T, '.')
# %%
plt.plot(v_sol_mu[0:9, :].T, '.')
# %%

fig_q, ax_q = plt.subplots(figsize=(9, 9))
ax_q.plot(u_sol_mu[0:5, :].T, v_sol_mu[0:5, :].T, '.', mec='0.7',  ls='')
# ax_q.set_xlim(x_lims)
# ax_q.set_ylim(y_lims)
plt.show()
# %%
for p_id in range(0, Np):
    for t_id in range(0, len(t_eval)):
        if x_sol_mu[p_id, t_id] < 0:
            u_sol_mu[p_id, t_id] = -uu_sol_mu[p_id, t_id]


plt.plot(v_sol_mu[0:4, :].T, u_sol_mu[0:4, :].T, '.')
# %%
plt.plot(u_sol_mu[0:20, :].T, v_sol_mu[0:20, :].T, '.')

# %% traj uv

u_lims = (-200, 200)
v_lims = (0, 18)

fig_q, ax_q = plt.subplots(figsize=(9, 9))
ax_q.quiver(uuu[::2, ::2], vvv[::2, ::2], ffx[::2, ::2],
            ffy[::2, ::2], color='gray', alpha=0.4)

# ax_q.pcolormesh(uuu[::2, ::2], vvv[::2, ::2], abs(Psi[::2, ::2]), alpha=.3, cmap = 'gray', shading='interp')
ax_q.plot(u_sol_mu[1:10, :].T, v_sol_mu[1:10, :].T, '.', mec='0.7',  ls='')
ax_q.set_title('Trajectories (20 Particles)')
ax_q.set_title(f'{ode_method}\n(ode = {ode_method})')
# ax_q.set_xlim(x_lims)
ax_q.set_ylim(v_lims)
plt.show()


# %%
fig, axes = plt.subplots(1, 5, figsize=(22, 6))

# Expanded limits so the final cloud is not cropped
x_lims = (-300, 300)
y_lims = (-800, 250)

# Create global CCD bins for the 1x5 plot so the "camera" resolution stays constant
pixel_size_global = 8
global_x_bins = np.arange(
    x_lims[0], x_lims[1] + pixel_size_global, pixel_size_global)
global_y_bins = np.arange(
    y_lims[0], y_lims[1] + pixel_size_global, pixel_size_global)

# Panel 0: Trajectories
ax_traj = axes[0]
ax_traj.quiver(X_mu[::2, ::2], Y_mu[::2, ::2], ffx[::2, ::2],
               ffy[::2, ::2], color='gray', alpha=0.4)
# ax_traj.add_patch(Rectangle((0, 0), 100, 100, linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
# ax_traj.add_patch(rect)
for i in range(100):
    ax_traj.plot(x_sol_mu[i, :], y_sol_mu[i, :], alpha=0.3,
                 linewidth=0.8, color='blue')
ax_traj.set_title('Trajectories (100 Particles)')

# Panels 1 to 4: CCD Density over time
time_indices = [0, len(t_eval)//3, 2*len(t_eval)//3, -1]
titles = ['Initial Time', 'Intermediate 1', 'Intermediate 2', 'Final Time']

for i, (ax, idx, title) in enumerate(zip(axes[1:], time_indices, titles)):
    current_x, current_y = x_sol_mu[:, idx], y_sol_mu[:, idx]

    # Use hist2d instead of hexbin
    _, _, _, im = ax.hist2d(current_x, current_y, bins=[
                            global_x_bins, global_y_bins], cmap='plasma', cmin=1)
    ax.add_patch(Rectangle((0, 0), 10, 10, linewidth=1,
                 edgecolor='red', facecolor='none', linestyle='--', alpha=0.5))
    ax.set_title(f'{title}\n(t = {t_eval[idx]*1e3:.2f}ms)')

# Formatting all 5 axes
for ax in axes:
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_aspect('equal')  # Keeps the aspect ratio accurate to real space
    ax.set_xlabel('X')
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Y')

fig.colorbar(im, ax=axes, label='Particles per Pixel', fraction=0.02, pad=0.02)
# %% plot trajectories
x_lims = (-200, 200)
y_lims = (-200, 150)
figq, ax_q = plt.subplots(figsize=(9, 9))
# ax_q.quiver(X_mu[::2, ::2], Y]_mu[::2, ::2], ffx[::2, ::2],
#             ffy[::2, ::2], color='gray', alpha=0.4)

ax_q.pcolormesh(X_mu[::2, ::2], Y_mu[::2, ::2], abs(Psi[::2, ::2]), alpha=.3)
# for i in range(20):
#     ax_q.plot(x_sol_mu[i, :], y_sol_mu[i, :],
#               alpha=0.3, linewidth=0.8)
ax_q.plot(x_sol_mu[1:10, :].T, y_sol_mu[1:10, :].T, '.', mec='0.7',  ls='')

ax_q.set_title('Trajectories (20 Particles)')
ax_q.set_title(f'{ode_method}\n(ode = {ode_method})')
ax_q.set_xlim(x_lims)
ax_q.set_ylim(y_lims)
plt.show()
# %%
plot_ccd_density(t_target=7e-3, x_sol=x_sol_mu, y_sol=y_sol_mu,
                 t_eval=t_eval, pixel_size=8)


# %% comparar ode methods

print("Solving ODEs...")
# solution = solve_ivp(particle_cloud_motion, (0, t_end), state0, t_eval=t_eval, method='RK45')
print(f"Solving ODEs for {Np} particles over {t_end} seconds...")

# %%
# t_end = 9.0e-3 #time in seconds
# t_eval = np.linspace(0, t_end, 200)
t_end = 11.0e-3  # time in seconds
t_span = (0, t_end)
t_eval = np.linspace(t_span[0], t_span[1], 200)

# %%
# Radau RK45 BDF LSODA
# ode_method_list = ['Radau', 'RK45', 'BDF', 'LSODA']
ode_method_list = ['RK45', 'LSODA']

for method in ode_method_list:
    print(method)

    start_time = time.perf_counter()
    ode_method = method
    solution = solve_ivp(particle_cloud_motion, t_span,
                         state0, t_eval=t_eval, method=ode_method)
    x_sol, y_sol = solution.y[0:Np, :], solution.y[Np:2*Np, :]

    # "toc"
    end_time = time.perf_counter()

    exec_time = end_time - start_time

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

    x_lims = (-200, 200)
    y_lims = (-200, 150)
    figq, ax_q = plt.subplots(figsize=(9, 9))
    # ax_q.quiver(X_mu[::2, ::2], Y]_mu[::2, ::2], ffx[::2, ::2],
    #             ffy[::2, ::2], color='gray', alpha=0.4)

    ax_q.pcolormesh(X_mu[::2, ::2], Y_mu[::2, ::2],
                    abs(Psi[::2, ::2]), alpha=.3)
    ax_q.plot(x_sol_mu[1:10, :].T, y_sol_mu[1:10, :].T,
              '.-', mec='0.7',  ls='')
    ax_q.set_title('Trajectories (20 Particles)')
    ax_q.set_title(f'exec_time{exec_time}s\n(ode = {ode_method})')
    ax_q.set_xlim(x_lims)
    ax_q.set_ylim(y_lims)
    plt.show()


# %%
u_lims = (-200, 200)
v_lims = (0, 18)

fig_q, ax_q = plt.subplots(figsize=(9, 9))
qq = 520
qq2 = 550
rr = 950
ax_q.quiver(uuu[qq:qq2, rr:-rr], vvv[qq:qq2, rr:-rr], ffx[qq:qq2, rr:-rr],
            ffy[qq:qq2, rr:-rr], color='gray', alpha=0.4)
# ax_q.pcolormesh(uuu, vvv, np.abs(Psi), alpha=.3, cmap = 'gray', shading='interp')
# ax_q.pcolormesh(uuu[qq:600, rr:-rr], vvv[qq:600, rr:-rr], np.abs(Psi[qq:600, rr:-rr]), alpha=.3, cmap = 'gray', shading='interp')
# ax_q.plot(u_sol_mu[1:10, :].T, v_sol_mu[1:10, :].T, '.', mec='0.7',  ls='')
# ax_q.set_title('Trajectories (20 Particles)')
# ax_q.set_title(f'{ode_method}\n(ode = {ode_method})')
# ax_q.set_xlim(x_lims)
# ax_q.set_ylim(v_lims)
plt.show()
