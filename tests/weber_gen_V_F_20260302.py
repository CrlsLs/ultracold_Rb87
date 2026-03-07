# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:37:54 2026

@author: clher
"""
import numpy as np
import mpmath
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import time
import pandas as pd

# Convert mpmath function to a vectorized numpy function
# This allows us to pass arrays to hyp1f1
hyp1f1_vec = np.frompyfunc(
    lambda a, b, z: complex(mpmath.hyp1f1(a, b, z)), 3, 1)

timestamp = datetime.now().strftime("%H%M%S")  # time stamp for saving
# bmp_name = f"W_a{a_param}_kp{kp}_{timestamp}.bmp"

# %% --- Parameters ---
um, vm = 14, 14
us = 0.02
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
kp = 1.0                # transversal part of wave vector
a_param = 5.0          # parameter "a" for Weber beam
np_val = 3.0

# --- Confluent Hypergeometric Functions ---
zu = 1j * kp * u**2
zu_noneg = 1j * kp * u_non_neg**2
zv = 1j * kp * v**2

vars0 = dict(u=u.copy(), v=v.copy(), a=a_param, kp=kp)

# %%
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

# %%
Ue_nn, Uo_nn, Ve, Vo = F11Ue_nn.copy(), F11Uo_nn.copy(), F11Ve.copy(), F11Vo.copy()
size_Uneg_start = len(uneg) - len(u_non_neg)
Ueneg = Ue_nn[size_Uneg_start:][::-1]
Uoneg = Uo_nn[size_Uneg_start:][::-1]
Ue = np.concatenate([Ueneg, Ue_nn])
Uo = np.concatenate([Uoneg, Uo_nn])

phase_u = np.exp(-zu/2)
phase_v = np.exp(-zv/2)

Vo = Vo*v  # **1.375
Uo = Uo*abs(u)  # **1.375
# Ve = Ve*phase_v*phase_u
# %%

UVs = ['Ue', 'Uo', 'Ve', 'Vo']
print("Generating Dual-Axis Plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes_flat = axes.flatten()

for i, func_name in enumerate(UVs):
    ax1 = axes_flat[i]
    data = locals()[func_name]

    # Plot Real Part on Left Axis
    line1 = ax1.plot(np.real(data), 'b-', label='Real', alpha=0.8)
    ax1.set_ylabel('Real Part', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create Second Axis for Imaginary Part
    ax2 = ax1.twinx()
    line2 = ax2.plot(np.imag(data), 'r.-', label='Imag', alpha=0.6)
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
# --- SLM Mapping ---
tel_sc = .15
yy = np.linspace(-1920 * tel_sc / 2, 1920 * tel_sc / 2, 1920)
xx = np.linspace(-1080 * tel_sc / 2, 1080 * tel_sc / 2, 1080)
# Note: xx is vertical, yy is horizontal, so we exchange them
yyy, xxx = np.meshgrid(yy, xx)

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
weight_odd = 1
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
# img.save(bmp_name)
# print(f"Hologram saved as {bmp_name}")


# --- Plot ---
plt.figure(221)
plt.imshow(np.abs(psi_epio2n)**2,
           extent=[yy.min(), yy.max(), xx.min(), xx.max()], cmap='jet')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()


figure(222)
plt.imshow(phase_map,
           extent=[yy.min(), yy.max(), xx.min(), xx.max()], cmap='gray')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()


m, n = psi_epio2.shape

# spatial sampling intervals (set to 1 if unknown)
dx = 0.8
dy = 0.8

# 1D frequency axes (after fftshift, monotonically increasing)
kx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))  # length n, along columns
ky = np.fft.fftshift(np.fft.fftfreq(m, d=dy))  # length m, along rows

# 2D grids
KX, KY = np.meshgrid(kx, ky)  # both shape (m, n)

# FFT
fftpsi = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psi_epio2)))
power = np.abs(fftpsi)**2

# Plot with physical axes
fig, ax = plt.subplots()
c = ax.pcolormesh(KX[400:-400, 700:-700], KY[400:-400, 700:-700],
                  (power[400:-400, 700:-700]), shading='gouraud', cmap='hot')
plt.colorbar(c, ax=ax)
ax.set_aspect('equal')
ax.set_xlabel('kx (cycles/unit)')
ax.set_ylabel('ky (cycles/unit)')
plt.show()


# %%
plt.figure(223)
[UU, VV] = np.meshgrid(u, v)
plt.pcolormesh(np.abs(psie))
# plt.colormap('jet')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()

figure(224)
plt.imshow(np.angle(psie), cmap='gray')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()
# %%
[UUo, VVo] = np.meshgrid(Uoc, Vo)
# %%
figure(235)
plt.pcolormesh(UU, VV, np.angle(VVo*VVo*UUo*UUo))
plt.title(f"Weber Beam Intensity (a={a_param})")
# plt.colorbar()
plt.show()

# %%
# %%
# FFT with shift
fftpsi = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psi_epio2)))

# Plot
fig, ax = plt.subplots()
power = np.abs(fftpsi[400:-400, 700:-700])**2

# pcolor equivalent with interpolation (pcolormesh with shading='gouraud' ≈ shading interp)
c = ax.pcolormesh((power), shading='gouraud', cmap='hot')
plt.colorbar(c, ax=ax)
ax.set_aspect('equal')
plt.show()

# %%
m, n = psi_epio2.shape

# spatial sampling intervals (set to 1 if unknown)
dx = 0.8
dy = 0.8

# 1D frequency axes (after fftshift, monotonically increasing)
kx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))  # length n, along columns
ky = np.fft.fftshift(np.fft.fftfreq(m, d=dy))  # length m, along rows

# 2D grids
KX, KY = np.meshgrid(kx, ky)  # both shape (m, n)

# FFT
fftpsi = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psi_epio2)))
power = np.abs(fftpsi)**2

# Plot with physical axes
fig, ax = plt.subplots()
c = ax.pcolormesh(KX[400:-400, 700:-700], KY[400:-400, 700:-700],
                  (power[400:-400, 700:-700]), shading='gouraud', cmap='hot')
plt.colorbar(c, ax=ax)
ax.set_aspect('equal')
ax.set_xlabel('kx (cycles/unit)')
ax.set_ylabel('ky (cycles/unit)')
plt.show()

# %% Crop Psi
m, n = psi_epio2n.shape
crf = .45  # .477  # crop factor #to crop psi original
mc = round(m*crf)
nc = round(n*crf)
Psi_crop = psi_epio2n[mc:-mc, nc:-nc]  # psi cropped
print(Psi_crop.shape)

figure(555)
plt.quiver(np.real(Psi_crop), np.imag(Psi_crop), width=0.0003)
