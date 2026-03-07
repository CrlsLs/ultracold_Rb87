# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 21:51:57 2026

@author: clher
"""
# %% Load packages
import numpy as np
import mpmath
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import time
# %% Convert mpmath function to a vectorized numpy function
# This allows us to pass arrays to hyp1f1
hyp1f1_vec = np.frompyfunc(
    lambda a, b, z: complex(mpmath.hyp1f1(a, b, z)), 3, 1)

# %% --- Parameters ---
um, vm = 11, 11
us = 0.03
vs = us  # / 2
# u = np.arange(0, um + us, us)
# v = np.arange(0, vm + vs, vs)
u = np.linspace(-um, um, int(um/us))
v = np.linspace(0, vm, int(vm/vs))

# uneg = -u[1:][::-1]

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
start_time = time.perf_counter()

F11Ue = hyp1f1_vec(0.25 - 0.5j * a_param, 0.5, zu).astype(complex)
F11Uo = hyp1f1_vec(0.75 - 0.5j * a_param, 1.5, zu).astype(complex)

end_time = time.perf_counter()
print(end_time-start_time)

start_time = time.perf_counter()

# V components: a = 1/4 + i*a/2 or 3/4 + i*a/2
F11Ve = hyp1f1_vec(0.25 + 0.5j * a_param, 0.5, zv).astype(complex)
F11Vo = hyp1f1_vec(np_val/4 + 0.5j * a_param, np_val/2, zv).astype(complex)

end_time = time.perf_counter()
print(end_time-start_time)

print("Computing hypergeometric functions finished")

# %%
Ue, Uo, Ve, Vo = F11Ue.copy(), F11Uo.copy(), F11Ve.copy(), F11Vo.copy()
# %%
# %%
UVs = ['Ue', 'Uo', 'Ve', 'Vo']
print("Generating Dual-Axis Plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes_flat = axes.flatten()

for i, func_name in enumerate(UVs):
    ax1 = axes_flat[i]
    data = globals()[func_name]

    # Plot Real Part on Left Axis
    line1 = ax1.plot(np.real(data), 'b-', label='Real', alpha=0.8)
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
