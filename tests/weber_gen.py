# %% Load packages
"""
Created on Thu Feb 26 21:59:31 2026
@author: clher
"""

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
um, vm = 15, 15
us = 0.03
vs = us / 2
# u = np.arange(0, um + us, us)
# v = np.arange(0, vm + vs, vs)
u = np.linspace(0, um, int(um/us))
v = np.linspace(0, vm, int(vm/vs)-1)

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
start_time = time.perf_counter()

F11Ue = hyp1f1_vec(0.25 - 0.5j * a_param, 0.5, zu).astype(complex)
F11Uo = hyp1f1_vec(0.75 - 0.5j * a_param, 1.5, zu).astype(complex)

end_time = time.perf_counter()
print(end_time-start_time)

start_time = time.perf_counter()

# V components: a = 1/4 + i*a/2 or 3/4 + i*a/2
F11Ve = hyp1f1_vec(0.25 + 0.5j * a_param, 0.5, zv).astype(complex)
F11Vo = hyp1f1_vec(0.75 + 0.5j * a_param, 1.5, zv).astype(complex)

end_time = time.perf_counter()
print(end_time-start_time)

print("Computing hypergeometric functions finished")


# %% This part is just to analize the behavior of U and V functions
Ue, Uo, Ve, Vo = F11Ue.copy(), F11Uo.copy(), F11Ve.copy(), F11Vo.copy()

# Create symmetric arrays for U
Ueneg = Ue[1:][::-1]
Uoneg = Uo[1:][::-1]
Uec = np.concatenate([Ueneg, Ue])
Uoc = np.concatenate([Uoneg, Uo])
uc = np.concatenate([uneg, u])
# %%
# %% --- Phase and Merge ---
phase_u = np.exp(1j * kp * u**2 / 2)
phase_uc = np.exp(1j * kp * uc**2 / 2)
phase_v = np.exp(1j * kp * v**2 / 2)

scale_factor = np.sqrt(uc**2+v**2)

factorF11Uec = (kp*uc**2)**(1-1/4)/scale_factor
factorF11Uoc = (kp*uc**2)**(3-1/4)/scale_factor

factorF11Ve = (kp*v**2)**(1-1/4)/scale_factor
factorF11Vo = (kp*v**2)**(3-1/4)/scale_factor
# %%
# Uec *= factorF11Uec
# Uoc *= factorF11Uoc
# Ve *= factorF11Vo
# Vo *= factorF11Ve

# Uec *= phase_uc
# Uoc = np.sqrt(kp) * uc * Uoc * phase_uc
# Ve *= phase_v
# Vo = np.sqrt(kp) * v * Vo * phase_v

# Uec = Uec * phase_uc
# Uoc = Uoc * np.sqrt(kp) * uc * phase_uc
# Ve = Ve * phase_v
# Vo = Vo * np.sqrt(kp) * v * phase_v

# Uec = Uec * phase_uc
# Uo = Uo * np.sqrt(kp) * u * phase_u
# Uoneg = -Uo[1:][::-1]
# Uoc = np.concatenate([Uoneg, Uo])
# Ve = Ve * phase_v
# Vo = Vo * np.sqrt(kp) * v * phase_v

Uec = Uec * phase_uc
Uoc = Uoc * phase_uc
Ve = Ve * phase_v
Vo = Vo * phase_v


# %% This part is just to analize the behavior of U and V functions
zuneg = 1j * kp * uneg**2

F11Ueneg_pr = hyp1f1_vec(0.25 - 0.5j * a_param, 0.5, zuneg).astype(complex)
F11Uoneg_pr = hyp1f1_vec(0.75 - 0.5j * a_param, 1.5, zuneg).astype(complex)

# %%
# Instead of separate variables:
UV_funcs = {
    'Uec': Uec,
    'Uoc': Uoc,
    'Ve': Ve,
    'Vo': Vo
}

print("Generating Plots...")

# Note: axes is a 2D array if you use (2, 2), so we flatten it for the loop
fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=False, sharey=False)
axes_flat = axes.flatten()

for i, (name, data) in enumerate(UV_funcs.items()):
    ax = axes_flat[i]

    ax.plot(np.real(data))
    ax.set_title(f"Real part of {name}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
UVs = ['Uec', 'Uoc', 'Ve', 'Vo']
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

# %%
UVs = ['Uoc', 'Uec', 'Vo', 'Ve']
print("Generating Plots...")

# Note: axes is a 2D array if you use (2, 2), so we flatten it for the loop
fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=False, sharey=False)
axes_flat = axes.flatten()

for i, func_name in enumerate(UVs):
    ax = axes_flat[i]

    actual_data = globals()[func_name]

    ax.plot(np.real(actual_data))
    ax.set_title(f"Real part of {func_name}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
figure(333)
plt.plot(np.real(F11Ve))
# %%
plt.plot(np.real(F11Uoneg_pr))

# %%
# --- SLM Mapping ---
tel_sc = .1
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
psie = np.outer(Uec, Ve)
psio = np.outer(Uoc, Vo)
psi_epio = (psie + 1j*np.pi * psio)  # *phase_uc*phase_v

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
# img.save(bmp_name)
# print(f"Hologram saved as {bmp_name}")

# %%
# --- Plot ---
plt.figure(221)
plt.imshow(np.abs(psi_epio2n)**2,
           extent=[yy.min(), yy.max(), xx.min(), xx.max()], cmap='jet')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()

# %%
figure(222)
plt.imshow(phase_map,
           extent=[yy.min(), yy.max(), xx.min(), xx.max()], cmap='gray')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()

# %%
plt.figure(243)
[UU, VV] = np.meshgrid(uc, v)
plt.pcolormesh(np.abs(psio))
# plt.colormap('jet')
plt.title(f"Weber Beam Intensity (a={a_param})")
plt.colorbar()
plt.show()
# %%
figure(225)
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
# FFT with shift
fftpsi = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psi_epio2)))

# Plot
fig, ax = plt.subplots()
power = np.abs(fftpsi)**2

# pcolor equivalent with interpolation (pcolormesh with shading='gouraud' ≈ shading interp)
c = ax.pcolormesh(power, shading='gouraud', cmap='hot')
plt.colorbar(c, ax=ax)
ax.set_aspect('equal')
plt.show()

# %% Crop Psi
m, n = psi_epio2n.shape
crf = .41  # .477  # crop factor #to crop psi original
mc = round(m*crf)
nc = round(n*crf)
Psi_crop = psi_epio2n[mc:-mc, nc:-nc]  # psi cropped
print(Psi_crop.shape)
# %%
figure(555)
plt.quiver(np.real(Psi_crop), np.imag(Psi_crop), width=0.0007)

# %%
m, n = psi.shape

# spatial sampling intervals (set to 1 if unknown)
dx = 1.0
dy = 1.0

# 1D frequency axes (after fftshift, monotonically increasing)
kx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))  # length n, along columns
ky = np.fft.fftshift(np.fft.fftfreq(m, d=dy))  # length m, along rows

# 2D grids
KX, KY = np.meshgrid(kx, ky)  # both shape (m, n)

# FFT
fftpsi = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psi)))
power = np.abs(fftpsi)**2

# Plot with physical axes
fig, ax = plt.subplots()
c = ax.pcolormesh(KX, KY, power, shading='gouraud', cmap='hot')
plt.colorbar(c, ax=ax)
ax.set_aspect('equal')
ax.set_xlabel('kx (cycles/unit)')
ax.set_ylabel('ky (cycles/unit)')
plt.show()
