# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 14:26:02 2026

@author: clher
"""
# %%
from types import SimpleNamespace
import pickle
from datetime import datetime
import json
from scipy.interpolate import RegularGridInterpolator
import matplotlib.patches as patches
import argparse
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import k as k_B, hbar, m_u
from scipy.optimize import curve_fit
from scipy.stats import norm
import pandas as pd


# %%

fig11, ax11 = plt.subplots()

# plt.scater(all_positions)
ax11.scatter(positions[:, 0], positions[:, 1], alpha=0.1)

# %%
# %plt.plot(trajectory[:,0,0])
plt.plot(times[:99], trajectory[:99, 3:9, 0], '.-')

# %%
plt.plot(trajectory[99, 3:9, 0], '.-')

# %%
plt.plot(x_fit[:]*1e6, trajectory[:200, 3:9, 0], '.-')

# %%
plt.plot(y_fit, '.-')

# %%
plt.plot(trajectory[0, :500, 0]*1e6,
         trajectory[0, :500, 1]*1e6, '.', alpha=.1)

# %%
plt.plot(trajectory[0, :500, 0]*1e6,
         trajectory[0, :500, 2]*1e6, '.', alpha=.1)

# %%
plt.scater(trajectory[180, :5000, 0]*1e6,
           trajectory[180, :5000, 1]*1e6, '.', alpha=.1)
# %%
a = np.array([1, 1, 3, 5, 8, 13])
# %%
b = np.array([(1, 2, 3), (4, 5, 6)], dtype=float)
c1 = np.array([[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (10, 11, 12)]], dtype=float)

# %%
c2 = np.array([[(1, 2), (3, 4), (5, 6)], [
              (7, 8), (9, 10), (11, 12)]], dtype=float)

# %%
c3 = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)],
              [(9, 10), (11, 12)]], dtype=float)

# %%
c232 = np.array([[(0, 1), (2, 3), (4, 5)], [
                (6, 7), (8, 9), (10, 11)]], dtype=float)

# %%
# Generate all triads (i, j, k) for i in [0,1], j in [0,1,2,3], k in [0,1]
c232b = np.zeros((2, 3, 2))
triads = []
l = 0
for i in range(2):          # i = 0, 1
    for j in range(3):      # j = 0, 1, 2, 3
        for k in range(2):  # k = 0, 1
            triads.append((i, j, k))
            c232b[i, j, k] = l
            l += 1
# Print the resulting sequence
for triad in triads:
    print(triad)
print(c232b)

# %%
c232 = np.arange(12).reshape(2, 3, 2)
print(c232)

# %%
c322 = np.arange(12).reshape(3, 2, 2)
print(c322)
# %%

triads = list(product(range(2), range(2), range(3)))
for triad in triads:
    print(triad)
# %%
fig = plt.figure()

# %%
fig2 = plt.figure(figsize=plt.figaspect(2.0))
# %%
fig.add_axes()
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(212)
fig3, axes = plt.subplots(nrows=2, ncols=2)

# %%
for row, res_idx in enumerate(plot_times_idx):
    print(row)
    print(res_idx)
# %%
parser = argparse.ArgumentParser(
    description='Simulate Weber forces on ultracold Rb-87 atomic cloud',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# %%
# script.py

# 1. Create the ArgumentParser object
parser = argparse.ArgumentParser(description="A simple script using argparse.")

# 2. Add arguments
parser.add_argument("name", help="The name of the user.")
parser.add_argument("--greet", action="store_true", help="Include a greeting.")
# %%

# 3. Parse the arguments from the command line
args = parser.parse_args('bob')

# 4. Access and use the arguments in your program
if args.greet:
    print(f"Hello, {args.name}!")
else:
    print(f"{args.name}, no greeting for you.")

# %%

# ---------------------------------------------------------
# 1. Setup Dummy Data (Your original m x n grid)
# ---------------------------------------------------------
x_sys_coord = np.linspace(0, 5, 10)  # n columns
y_sys_coord = np.linspace(0, 4, 8)   # m rows
X_orig, Y_orig = np.meshgrid(x_sys_coord, y_sys_coord)

# Dummy force field (swirling pattern for visibility)
ffx = np.cos(X_orig) * np.sin(Y_orig)
ffy = np.sin(X_orig) * np.cos(Y_orig)

# Your exact interpolator setup
interp_fx = RegularGridInterpolator(
    (y_sys_coord, x_sys_coord), ffx, bounds_error=False, fill_value=0.0)
interp_fy = RegularGridInterpolator(
    (y_sys_coord, x_sys_coord), ffy, bounds_error=False, fill_value=0.0)

# ---------------------------------------------------------
# 2. Create the Extended Grid
# ---------------------------------------------------------
# Make the limits significantly wider than the original 0-5 and 0-4 range
x_ext = np.linspace(-2, 7, 25)
y_ext = np.linspace(-2, 6, 25)
X_ext, Y_ext = np.meshgrid(x_ext, y_ext)

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

# Plot the vector field
ax.quiver(X_ext, Y_ext, ffx_ext, ffy_ext, color='blue', pivot='middle')

# Draw a red rectangle to show the original m x n boundary
rect = patches.Rectangle(
    (x_sys_coord.min(), y_sys_coord.min()),           # (x, y) bottom-left corner
    x_sys_coord.max() - x_sys_coord.min(),            # Width
    y_sys_coord.max() - y_sys_coord.min(),            # Height
    linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Original Boundary'
)
ax.add_patch(rect)

# Formatting
ax.set_title("Force Field Interpolation (Zero Outside Boundary)")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.axis('equal')  # Keeps the aspect ratio square so vectors point accurately
plt.show()

# %%
# 2. Create the Extended Grid
# ---------------------------------------------------------
# Make the limits significantly wider than the original 0-5 and 0-4 range
w_factor = 1.2  # widering factor
x_ext = (np.arange(nx*w_factor) - nx*w_factor/2) * dx
y_ext = (np.arange(ny*w_factor) - ny*w_factor/2) * dy
X_ext, Y_ext = np.meshgrid(x_ext, y_ext)

# ---------------------------------------------------------
# 3. Evaluate Interpolators on Extended Grid
# ---------------------------------------------------------
# Stack points into (N, 2).
# Note: Because your interpolator takes (y, x), we stack Y_ext then X_ext
eval_points = np.column_stack((Y_ext.ravel(), X_ext.ravel()))

# Evaluate and reshape back to the 2D grid shape for plotting
ffx_ext = interp_ax(eval_points).reshape(X_ext.shape)
ffy_ext = interp_ay(eval_points).reshape(X_ext.shape)

# ---------------------------------------------------------
# 4. Plot the Results
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the vector field
ax.quiver(X_ext, Y_ext, ffx_ext, ffy_ext, color='blue', pivot='middle')

# Draw a red rectangle to show the original m x n boundary
rect = patches.Rectangle(
    (x_sys_coord.min(), y_sys_coord.min()),           # (x, y) bottom-left corner
    x_sys_coord.max() - x_sys_coord.min(),            # Width
    y_sys_coord.max() - y_sys_coord.min(),            # Height
    linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Original Boundary'
)
ax.add_patch(rect)

# Formatting
ax.set_title("Force Field Interpolation (Zero Outside Boundary)")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.axis('equal')  # Keeps the aspect ratio square so vectors point accurately
plt.show()

# %% json file load

data = {
    "timestamp": datetime.now(),
    "values": (1, 2, 3)          # tuple
}


def convert(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()   # convert datetime to string
    if isinstance(obj, tuple):
        return list(obj)         # convert tuple to list
    raise TypeError(f"Type {type(obj)} not serializable")


with open(r"..\data\processed\20260324_from_colab\data.json", "w") as file:
    json.dump(data, file, default=convert)   # custom serializer

# On loading, you'd need to convert back (e.g., parse datetime strings)
# %%

with open(r"..\data\processed\20260324_from_colab\data.json", "r") as file:
    loaded_data = json.load(file)  # read JSON back into a dictionary

print(loaded_data)

# %% save picke

data = {
    "name": "Alice",
    "age": 30,
    "skills": {"Python", "Data Analysis"},  # set is allowed
    "active": True
}

# note "wb" for binary write
with open(r"..\data\processed\20260324_from_colab\data.pkl", "wb") as file:
    pickle.dump(data, file)

# %% load picke

path = r'..\data\processed\20260324_from_colab\data.pkl'

with open(path, 'rb') as f:
    data_pkl_loaded = pickle.load(f)

# %%
filename = "hypergeometric_results_colab.pkl"
filePath = data_path / filename


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_ns(x) for x in d]
    else:
        return d


with open(filePath, 'rb') as f:
    raw = pickle.load(f)

results = dict_to_ns(raw)  # preserves structure, just changes access style
# %%
listw = []
data = {"name": "Alice", "age": 30, 'w': listw}
# %%
data.update({"w": listw.append(33)})

print(data)
# Output: {'name': 'Alice', 'age': 30, 'city': 'New York', 'active': True}

# %%
hypergeometric_results.update({
    'parameters': {
        'a_param': a_param,
        'kp': kp},
    'hypergf': {
        'F11Ue_nn': F11Ue_nn.copy(),
        'F11Uo_nn': F11Uo_nn.copy(),
        'F11Ve': F11Ve.copy(),
        'F11Vo': F11Vo.copy()}
})
