# %% Load pkgs-*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:49:27 2026

@author: clher
"""
"""
This script makes the Weber profiles with hypergeometric functions in parabolic 
coordinates, then it is interpolated to transform to rectangular coordinates.
Then it simulates the fall of Rb87 in the force field defined by the Weber beam
and gravity.
Pay attention to the exchange $x$ -> $y$, to have $x$ vertical. Or letting $x$ be horizontal and changing the coordinates in the Rb87 cloud. 
Based on Hernández-Cedillo C.L. et al. Phys. Rev. A, 87 2013
"""


# Convert mpmath function to a vectorized numpy function
# This allows us to pass arrays to hyp1f1
# import pandas as pd

# Convert mpmath function to a vectorized numpy function
# This allows us to pass arrays to hyp1f1


# %% Define functions




import json
from pathlib import Path
import scipy.io
from scipy import constants
from scipy.integrate import solve_ivp
import numpy as np
import mpmath
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import time
import pandas as pd
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pickle  # load pkl files
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

    F_rad_coeff = (hbar * Gamma / 2) * p_0  # *10 for debugging
    F_rad_x = F_rad_coeff * phase_grad_x / denom
    F_rad_y = F_rad_coeff * phase_grad_y / denom

    # Accelerations
    a_x = (F_grad_x + F_rad_x) / m_Rb87  # *.2 for debugging
    a_y = (F_grad_y + F_rad_y) / m_Rb87 - 9.8  # gravity added

    print(['ax_phase', 'ay_phase', 'ax_I', 'ay_I'], 'maximos')
    print([np.max(F_rad_x) / m_Rb87, np.max(F_rad_y) / m_Rb87],
          [np.max(F_grad_x) / m_Rb87, np.max(F_grad_x) / m_Rb87])

   # --- TABLE DISPLAY LOGIC ---
    data = [
        ("Omega_0 (Resonance)", f"{omega_0:.4e}", "rad/s"),
        ("Omega_L (Laser)", f"{omega_L:.4e}", "rad/s"),
        ("Delta Omega (Detuning)", f"{delta_omega:.4e}", "rad/s"),
        ("g (Coupling)", f"{g:.4e}", "Hz"),
        ("p_0 (Saturation)", f"{p_0:.4e}", "Unitless"),
        ("Max Intensity (I)", f"{np.max(I):.4e}", "Normalized"),
        ("F_grad_coeff", f"{F_grad_coeff:.4e}", "N*m"),
        ("F_rad_coeff", f"{F_rad_coeff:.4e}", "N*m/rad"),
        ("Max Acceleration X", f"{np.max(np.abs(a_x)):.4e}", "m/s^2")
    ]

    print("\n" + "="*50)
    print(f"{'Quantity':<25} | {'Value':<12} | {'Unit'}")
    print("-" * 50)
    for name, val, unit in data:
        print(f"{name:<25} | {val:<12} | {unit}")
    print("="*50 + "\n")
    # ---------------------------

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
    actual_t = t_eval[idx]*1e3
    Np = x_sol.shape[0]

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
        x, y, bins=[x_bins, y_bins], cmap='jet', cmin=1)

    # Draw the boundary of the force field if it falls in the camera view
    rect_ref = Rectangle((0, 0), 10, 10, linewidth=1.5,
                         edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect_ref)

    fig.colorbar(im, ax=ax, label='Particles per Pixel')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')  # Forces pixels to be perfectly square

    ax.set_title(f't = {actual_t:.1f}ms, a={
                 a_param}, Np = {Np}, ODE={ode_method}')
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


# %% random seeds
rng = np.random.default_rng(42)
# random_numbers = rng.random(5)

# Method 2: Legacy approach (Older code)
np.random.seed(42)
# random_numbers_legacy = np.random.rand(5)

hyp1f1_vec = np.frompyfunc(
    lambda a, b, z: complex(mpmath.hyp1f1(a, b, z)), 3, 1)

timestamp = datetime.now().strftime("%H%M%S")  # time stamp for saving
# bmp_name = f"W_a{a_param}_kp{kp}_{timestamp}.bmp"

hypergeometric_results = []  # Initialize list to store results

# %% Weber gen--- Parameters ---
% % timeit
readme = "kp =1, since it only re-scale the spatial coordinates. The first block use units of pixel size for distance units."
um, vm = 16, 16
us = 0.05
vs = us  # / 2
# u = np.arange(0, um + us, us)
# v = np.arange(0, vm + vs, vs)
u = np.linspace(-um, um, int(um/us))
v = np.linspace(0, vm, int(vm/vs))
uneg = u[u < 0]  # -u[1:][::-1]
u_non_neg = u[u >= 0]
# uc = np.concatenate([uneg, u])

sp = 8.0                # pixel size (microns)
lambd = 0.782 / sp      # wavelength
kp = 1.0                # transversal part of wave vector, to be re-escalated later as kp_pxsi
a_param = 5.0          # parameter "a" for Weber beam
np_val = 3.0

# --- Confluent Hypergeometric Functions ---
zu = 1j * kp * u**2
zu_noneg = 1j * kp * u_non_neg**2
zv = 1j * kp * v**2

# vars0 = dict(u=u.copy(), v=v.copy(), a=a_param, kp=kp)
params = dict(a=a_param, kp=kp, readme=readme, lambd=lambd,
              sp=sp, um=um, vm=vm, us=us, vs=vs)
sparams = pd.Series(params)
# %% hypergeometric functions
print("Computing hypergeometric functions (this may take a moment)...")
# U components: a = 1/4 - i*a/2 or 3/4 - i*a/2
start_time = time.perf_counter()

F11Ue_nn = hyp1f1_vec(0.25 - 0.5j * a_param, 0.5, zu_noneg).astype(complex)
F11Uo_nn = hyp1f1_vec(0.75 - 0.5j * a_param, 1.5, zu_noneg).astype(complex)

end_time = time.perf_counter()
print(end_time-start_time)

start_time = time.perf_counter()

# V components: a = 1/4 + i*a/2 or 3/4 + i*a/2
F11Ve = hyp1f1_vec(0.25 + 0.5j * a_param, 0.5, zv).astype(complex)
F11Vo = hyp1f1_vec(np_val/4 + 0.5j * a_param, np_val/2, zv).astype(complex)

end_time = time.perf_counter()
print(end_time-start_time)

print("Computing hypergeometric functions finished")

# %% Or load from G Colab data
# F11Ue_nn_GColab = np.load('C:/Users/clher/Documents/programacion/projects/ultracold_Rb87/data/processed/20260324_from_colab/F11Ue_nn.npy')

data_path = Path(
    r"..\data\processed\20260324_from_colab")

# Psi_path = folder_path / 'Psi_a10_kp3.mat'
# mat_data = scipy.io.loadmat(Psi_path)  # load matlab mat file
# Psi_or = mat_data['Psi']  # psi original
# df = pd.DataFrame(Psi_or)
#
# Psi = df.map(safe_complex).values if hasattr(
#     df, 'map') else df.applymap(safe_complex).values

# load picke
filename = "hypergeometric_results_colab_a0,5,10,20,40,us-0p01.pkl"
filePath = data_path / filename

with open(filePath, 'rb') as f:  # "rb" for binary
    dataloaded = pickle.load(f)

# data = dataloaded[0]
# %%
entry = dataloaded[0]

for key, value in entry.items():
    globals()[key] = value


Ue_nn, Uo_nn, Ve, Vo = F11Ue_nn.copy(), F11Uo_nn.copy(), F11Ve.copy(), F11Vo.copy()
size_Uneg_start = len(uneg) - len(u_non_neg)
Ueneg = Ue_nn[size_Uneg_start:][::-1]
Uoneg = Uo_nn[size_Uneg_start:][::-1]
Ue = np.concatenate([Ueneg, Ue_nn])
Uo = np.concatenate([Uoneg, Uo_nn])
# %%
phase_u = np.exp(-zu/2)
phase_v = np.exp(-zv/2)

Vo = Vo*v  # **1.375
Uo = Uo*abs(u)  # *u
# Ve = Ve*phase_v*phase_u
# %% plots UV's

UVs = ['Ue', 'Uo', 'Ve', 'Vo']
uvs_coords = ['u', 'u', 'v', 'v']
print("Generating Dual-Axis Plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes_flat = axes.flatten()

for i, func_name in enumerate(UVs):
    ax1 = axes_flat[i]
    data = locals()[func_name]
    coords = locals()[uvs_coords[i]]

    # Plot Real Part on Left Axis
    line1 = ax1.plot(coords, np.real(data), 'b-', label='Real', alpha=0.8)
    ax1.set_ylabel('Real Part', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create Second Axis for Imaginary Part
    ax2 = ax1.twinx()
    line2 = ax2.plot(coords, np.imag(data), 'r.-', label='Imag', alpha=0.6)
    ax2.set_ylabel('Imaginary Part', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_title(f"Weber Function: {func_name}")
    ax1.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

print("Generating Dual-Axis Plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes_flat = axes.flatten()

for i, func_name in enumerate(UVs):
    ax1 = axes_flat[i]
    data = locals()[func_name]
    coords = locals()[uvs_coords[i]]

    # Plot Real Part on Left Axis
    line1 = ax1.plot(coords, np.abs(data), 'b.-', label='Real', alpha=0.8)
    ax1.set_ylabel('Abs Part', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create Second Axis for Imaginary Part
    ax2 = ax1.twinx()
    line2 = ax2.plot(coords, np.angle(data), 'r.-', label='Imag', alpha=0.6)
    ax2.set_ylabel('Angle Part', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_title(f"Weber Function: {func_name}")
    ax1.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# %% psi's
# --- SLM Mapping ---
tel_sc = 1

yy = np.linspace(-1920 * tel_sc / 2, 1920 * tel_sc /
                 2, 1920)  # limits for the SLM
xx = np.linspace(-1080 * tel_sc / 2, 1080 * tel_sc / 2, 1080)

# yy, xx in units of pixel size reduced
# limits to reprouce the paper figure
# limits for the paper image cropped
# yy = np.linspace(-670 * tel_sc / 2, 670 * tel_sc / 2, int(670/.8))
# xx = np.linspace(-420 * tel_sc / 2, 420 * tel_sc / 2, int(420/.8))

# Note: xx is vertical, yy is horizontal, so we exchange them
kp_pxsi = .3  # kp in inverse units of pixel size reduced .375*.8
yyy, xxx = np.meshgrid(yy*kp_pxsi, xx*kp_pxsi)

caux = xxx + 1j * yyy
uuuu = np.abs(np.real(np.sqrt(2 * caux)))
vvvv = np.abs(np.imag(np.sqrt(2 * caux)))

uuu = uuuu.copy()
mid_idx = uuu.shape[1] // 2
uuu[:, :mid_idx] = -uuuu[:, :mid_idx]

# --- Field Construction & Interpolation ---
psie = np.outer(Ue, Ve)
psio = np.outer(Uo, Vo)

phase_uv = np.outer(phase_u, phase_v)
weight_odd = 1*9
weight_even = np.pi
psi_epio = (weight_even*psie/np.max(np.abs(psie)) + weight_odd*1j*psio /
            np.max(np.abs(psio))) * phase_uv

# Interpolate onto the SLM grid
interp_func = RegularGridInterpolator(
    (u, v), psi_epio, bounds_error=False, fill_value=0)
pts = np.stack([uuu.ravel(), vvvv.ravel()], axis=-1)
psi_epio2 = interp_func(pts).reshape(uuu.shape)
psi_epio2n = psi_epio2 / np.max(np.abs(psi_epio2))

# --- Save Hologram as BMP ---
phase_map = np.angle(psi_epio2n)
# Normalize phase to 0-255 for 8-bit SLM
phase_scaled = ((phase_map - phase_map.min()) /
                (phase_map.max() - phase_map.min()) * 255).astype(np.uint8)

# Crop/Resize to exact SLM dimensions (1080, 1920)
img = Image.fromarray(phase_scaled[:1080, :1920])
timestamp = datetime.now().strftime("%H%M%S")
bmp_name = f"W_a{a_param}_kp{kp}_{timestamp}.bmp"
# %% --- Plot ---

# exchange xxx -> yyy, to have $x$ vertical

plt.figure(221)
plt.imshow(np.abs(psi_epio2n)**2,
           extent=[yy.min(), yy.max(), xx.min(), xx.max()], cmap='jet')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()

plt.figure(2211)
plt.pcolormesh(xxx, yyy, np.abs(psi_epio2n)**2,
               cmap='jet')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.axis('equal')
plt.show()


plt.figure(222)
plt.imshow(phase_map,
           extent=[yy.min(), yy.max(), xx.min(), xx.max()], cmap='gray')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()

# %%
npx_crop = 90
Psi = np.flipud(psi_epio2n[npx_crop:-1, :])
print(Psi.shape)

plt.pcolormesh(np.abs(Psi)**2)
plt.figure()
plt.pcolormesh(np.angle(Psi))


# %% Defining rectangular coords to be displayed on the SLM, and croping Weber
# profile to optimize the cloude simulation

pr, pc = psi_epio2n.shape
npx_crop = int(pr/2-140/.8)

Psi = np.flipud(psi_epio2n[npx_crop:-1, :])
print(Psi.shape)

plt.pcolormesh(np.abs(Psi)**2)
plt.figure()
plt.pcolormesh(np.angle(Psi))

# %% Defining rectangular coords to be displayed on the SLM, and croping Weber
# profile for the cloude simulation

nr, mc = Psi.shape  # nr-> num rows, mc -> num cols
tel_sc = .1  # telescope scale factor
kppx = 1  # in pixel size ^(-1)
pxs = 8  # pixel size in microns
fc_um = 1e-6  # to conver to microns
tscf = tel_sc*pxs*kppx*fc_um  # total scale factor

x_sys_coord = np.linspace(-mc * tscf / 2, mc * tscf / 2, mc)
y_sys_coord = np.linspace(-nr * tscf / 2,
                          (nr - npx_crop) * tscf / 2, nr)

# x_sys_coord = np.linspace(-670 * tel_sc / 2, 670 * tel_sc / 2, mc) #limits for the paper image cropped
# y_sys_coord = np.linspace(-420 * tel_sc / 2,
#                           (420 - npx_crop*.8*2) * tel_sc / 2, nr)

# Note: xx is vertical, yy is horizontal, so we exchange them
kp_pxsi = 1
X_grid, Y_grid = np.meshgrid(x_sys_coord*kp_pxsi, y_sys_coord*kp_pxsi)

print(X_grid.shape)
plt.figure()
plt.pcolormesh(X_grid, Y_grid, np.abs(Psi)**2)

# %% @title %% Get accelerations fields
dx = dy = 0.8e-6  # pixel size times telescope reduction
print("Calculating Field Accelerations...")
ffx, ffy, Intensity = get_weber_acceleration_grids(
    Psi, dx, dy)  # ffx,ffy are already accelerations, not force
print("accelerations, complete")
print(ffx.shape, ffy.shape, Intensity.shape)
# #%%
plt.figure()
plt.quiver(X_grid, Y_grid, ffx, ffy, Intensity)
plt.show()


# %% Interp acceleration fields

dx = dy = 0.8e-6  # pixel size times telescope reduction
ny, nx = Psi.shape
interp_fx = RegularGridInterpolator(
    (y_sys_coord, x_sys_coord), ffx, method='cubic', bounds_error=False, fill_value=0.0)
interp_fy = RegularGridInterpolator(
    (y_sys_coord, x_sys_coord), ffy, method='cubic', bounds_error=False, fill_value=-9.8)

# %%
# this part shows if the interpolations is behaving as expected

# 2. Create the Extended Grid
# # ---------------------------------------------------------
# Make the limits significantly wider than the original 0-5 and 0-4 range
w_factor = .3  # 1.03  # widering factor
extra_factor = 1.5
x_ext = (np.arange(nx*w_factor) - nx*w_factor/2*extra_factor) * dx
y_ext = (np.arange(ny*w_factor) - ny*w_factor/2*extra_factor) * dy
X_ext, Y_ext = np.meshgrid(x_ext, y_ext)
nc, mc = X_ext.shape
print(mc, nc)
# ---------------------------------------------------------
# 3. Evaluate Interpolators on Extended Grid
# ---------------------------------------------------------
# Stack points into (N, 2).
# Note: Because your interpolator takes (y, x), we stack Y_ext then X_ext
eval_points = np.column_stack((Y_ext.ravel(), X_ext.ravel()))

# Evaluate and reshape back to the 2D grid shape for plotting
ffx_ext = interp_fx(eval_points).reshape(X_ext.shape)
ffy_ext = interp_fy(eval_points).reshape(X_ext.shape)

# ---------------------------------------------------------
# 4. Plot the Results
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

col_lim = int(nc*.6)  # 60
row_lim = int(mc*.6)  # 100
# Plot the vector field
ax.quiver(X_ext[0:col_lim, 0:row_lim], Y_ext[0:col_lim, 0:row_lim], ffx_ext[0:col_lim, 0:row_lim], ffy_ext[0:col_lim, 0:row_lim],
          color='blue', pivot='middle', width=0.0005)

# Draw a red rectangle to show the original m x n boundary
# rect = patches.Rectangle(
#     (x_sys_coord.min(), y_sys_coord.min()),           # (x, y) bottom-left corner
#     x_sys_coord.max() - x_sys_coord.min(),            # Width
#     y_sys_coord.max() - y_sys_coord.min(),            # Height
#     linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Original Boundary'
# )
# ax.add_patch(rect)

# Formatting
ax.set_title("Force Field Interpolation (Zero Outside Boundary)")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
# ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.axis('equal')  # Keeps the aspect ratio square so vectors point accurately
plt.show()

# %% # 1. Simulation Setup, cloud init
# ---------------------------------------------------------
Np = 5000

# --- 2. Initialize Cloud ---
print("Initializing Atomic Cloud...")
x0, y0, vx0, vy0 = generate_initial_cloud(
    N_atoms=Np)  # Reduced to 1k for faster testing

# Pack initial conditions into a single 1D array
state0 = np.concatenate((x0, y0, vx0, vy0))
# %% #2 Time setup
# t_end = 9.0e-3 #time in seconds
# t_eval = np.linspace(0, t_end, 200)

t_end = 17.0e-3  # time in seconds
t_span = (0, t_end)
t_eval = np.linspace(t_span[0], t_span[1], 100)
# %%
# %% # 3. Solving ODEs
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

# %% Plot clouds
fig, axes = plt.subplots(1, 5, figsize=(22, 6))

# Expanded limits so the final cloud is not cropped
x_lims = (-350, 350)
y_lims = (-1250, 200)

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
for i in range(33):
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
                            global_x_bins, global_y_bins], cmap='jet', cmin=1)
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
#
# %% camerea shot at paper times
plot_ccd_density(t_target=7e-3, x_sol=x_sol_mu, y_sol=y_sol_mu,
                 t_eval=t_eval, pixel_size=8)

plot_ccd_density(t_target=12e-3, x_sol=x_sol_mu, y_sol=y_sol_mu,
                 t_eval=t_eval, pixel_size=8)
# %%

# %% save data
# note "wb" for binary write
# with open(r"..\data\processed\20260324_from_colab\hypergeometric_results_colab.pkl", "wb") as file:
#     pickle.dump(data, file)


# # load picke
# with open("data.pkl", "wb") as file:       # note "wb" for binary write
#     pickle.dump(data, file)
