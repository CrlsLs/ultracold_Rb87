# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 22:53:37 2026

@author: clher
"""

import numpy as np
import mpmath
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import time

# Convert mpmath function to a vectorized numpy function
# This allows us to pass arrays to hyp1f1
hyp1f1_vec = np.frompyfunc(
    lambda a, b, z: complex(mpmath.hyp1f1(a, b, z)), 3, 1)

timestamp = datetime.now().strftime("%H%M%S")  # time stamp for saving
# bmp_name = f"W_a{a_param}_kp{kp}_{timestamp}.bmp"
# %%
# --- Parameters ---
um, vm = 12, 12
us = 0.05
vs = us  # / 2
# u = np.arange(0, um + us, us)
# v = np.arange(0, vm + vs, vs)
# uneg = -u[1:][::-1]

u = np.linspace(-um, um, int(um/us))
v = np.linspace(0, vm, int(vm/vs))

sp = 8.0                # pixel size (microns)
lambd = 0.782 / sp      # wavelength
kp = 1.0                # transversal part of wave vector
a_param = 5.0          # parameter "a" for Weber beam
np_val = 3.0

# --- Confluent Hypergeometric Functions ---
zu = 1j * kp * u**2
zv = 1j * kp * v**2

print("Computing hypergeometric functions (this may take a moment)...")
start_time = time.perf_counter()

# U components: a = 1/4 - i*a/2 or 3/4 - i*a/2
Ue = hyp1f1_vec(0.25 - 0.5j * a_param, 0.5, zu).astype(complex)
Uo = hyp1f1_vec(np_val/4 - 0.5j * a_param, np_val/2, zu).astype(complex)

# V components: a = 1/4 + i*a/2 or 3/4 + i*a/2
Ve = hyp1f1_vec(0.25 + 0.5j * a_param, 0.5, zv).astype(complex)
Vo = hyp1f1_vec(np_val/4 + 0.5j * a_param, np_val/2, zv).astype(complex)
end_time = time.perf_counter()
print(end_time-start_time)

print("Computing hypergeometric functions finished")

# %%
Uec = Ue.copy()
Uoc = Uo.copy()
uc = u
# %%
# --- Phase and Merge ---
phase_u = np.exp(-1j * kp * u**2 / 2)
phase_v = np.exp(-1j * kp * v**2 / 2)
# %%
Ue *= phase_u
Uo = np.sqrt(kp) * u * Uo * phase_u
Ve *= phase_v
Vo = np.sqrt(kp) * v * Vo * phase_v

# Create symmetric arrays for U
Ueneg = Ue[1:][::-1]
Uoneg = Uo[1:][::-1]
Uec = np.concatenate([Ueneg, Ue])
Uoc = np.concatenate([Uoneg, Uo])
uc = np.concatenate([uneg, u])

# --- SLM Mapping ---
tel_sc = 0.1
xx = np.linspace(-1920 * tel_sc / 2, 1920 * tel_sc / 2, 1920)
yy = np.linspace(-1080 * tel_sc / 2, 1080 * tel_sc / 2, 1080)
yyy, xxx = np.meshgrid(yy, xx)  # Note: xx is vertical, yy is horizontal

caux = xxx + 1j * yyy
uuuu = np.abs(np.real(np.sqrt(2 * caux)))
vvvv = np.abs(np.imag(np.sqrt(2 * caux)))

uuu = uuuu.copy()
mid_idx = uuu.shape[1] // 2
uuu[:, :mid_idx] = -uuuu[:, :mid_idx]

# --- Field Construction & Interpolation ---
psie = np.outer(Uec, Ve)
psio = np.outer(Uoc, Vo)
psi_epio = psie + 1j * psio

# Interpolate onto the SLM grid
interp_func = RegularGridInterpolator(
    (uc, v), psi_epio, bounds_error=False, fill_value=0)
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
img.save(bmp_name)
print(f"Hologram saved as {bmp_name}")

# --- Plot ---
plt.imshow(np.abs(psi_epio2n)**2,
           extent=[yy.min(), yy.max(), xx.min(), xx.max()], cmap='magma')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()
# %%
# %%
figure(222)
plt.imshow(phase_map,
           extent=[yy.min(), yy.max(), xx.min(), xx.max()], cmap='gray')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()

# %%
# %%
UVs = ['Uec', 'Uoc', 'Ve', 'Vo']
print("Generating Dual-Axis Plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes_flat = axes.flatten()

for i, func_name in enumerate(UVs):
    ax1 = axes_flat[i]
    data = globals()[func_name]

    # Plot Real Part on Left Axis
    line1 = ax1.plot(np.real(data)*v, 'b-', label='Real', alpha=0.8)
    ax1.set_ylabel('Real Part', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create Second Axis for Imaginary Part
    ax2 = ax1.twinx()
    line2 = ax2.plot(np.imag(data), 'r--', label='Imag', alpha=0.6)
    ax2.set_ylabel('Imaginary Part', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_title(f"Weber Function: {func_name}")
    ax1.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
# %%
print("Generating Dual-Axis Plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes_flat = axes.flatten()

for i, func_name in enumerate(UVs):
    ax1 = axes_flat[i]
    data = globals()[func_name]

    # Plot Real Part on Left Axis
    line1 = ax1.plot(np.abs(data), 'b.-', label='Real', alpha=0.8)
    ax1.set_ylabel('Abs Part', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create Second Axis for Imaginary Part
    ax2 = ax1.twinx()
    line2 = ax2.plot(np.angle(data), 'r.-', label='Imag', alpha=0.6)
    ax2.set_ylabel('Angle Part', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_title(f"Weber Function: {func_name}")
    ax1.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()


# %%
# --- Parameters ---
um, vm = 10, 10
us = 0.05
vs = us / 2
u = np.arange(0, um + us, us)
v = np.arange(0, vm + vs, vs)
uneg = -u[1:][::-1]

sp = 8.0                # pixel size (microns)
lambd = 0.782 / sp      # wavelength
kp = 3.0                # transversal part of wave vector
a_param = 5.0          # parameter "a" for Weber beam
np_val = 3.0

# --- Confluent Hypergeometric Functions ---
zu = 1j * kp * u**2
zv = 1j * kp * v**2

print("Computing hypergeometric functions (this may take a moment)...")

# U components: a = 1/4 - i*a/2 or 3/4 - i*a/2
Ue = hyp1f1_vec(0.25 - 0.5j * a_param, 0.5, zu).astype(complex)
Uo = hyp1f1_vec(np_val/4 - 0.5j * a_param, np_val/2, zu).astype(complex)

# V components: a = 1/4 + i*a/2 or 3/4 + i*a/2
Ve = hyp1f1_vec(0.25 + 0.5j * a_param, 0.5, zv).astype(complex)
Vo = hyp1f1_vec(np_val/4 + 0.5j * a_param, np_val/2, zv).astype(complex)

# --- Phase and Merge ---
phase_u = np.exp(-1j * kp * u**2 / 2)
phase_v = np.exp(-1j * kp * v**2 / 2)
# %%
Ue *= phase_u
Uo = np.sqrt(kp) * u * Uo * phase_u
Ve *= phase_v
Vo = np.sqrt(kp) * v * Vo * phase_v
# %%
# Create symmetric arrays for U
Ueneg = Ue[1:][::-1]
Uoneg = Uo[1:][::-1]
Uec = np.concatenate([Ueneg, Ue])
Uoc = np.concatenate([Uoneg, Uo])
uc = np.concatenate([uneg, u])
