# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:57:34 2026

@author: clher
"""
"""
Fuction to generate parabolic functions for Weber beams.
"""
# %% @title Loading packs

# Convert mpmath function to a vectorized numpy function
# This allows us to pass arrays to hyp1f1

# %%
from pathlib import Path
import scipy.io
from scipy import constants
from scipy.integrate import solve_ivp
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import numpy as np
import mpmath
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import time
import pandas as pd
import json
import pickle
hyp1f1_vec = np.frompyfunc(
    lambda a, b, z: complex(mpmath.hyp1f1(a, b, z)), 3, 1)

PROJECT_ROOT = Path(__file__).parent.parent

# %% --- Parameters ---
readme = "kp =1, since it only re-scale the spatial coordinates. The first block use units of pixel size for distance units."
um, vm = 20, 20
us = 0.9  # 0.1 only to test, the final data with more prescition will be done in Colab
vs = us  # / 2
# u = np.arange(0, um + us, us)
# v = np.arange(0, vm + vs, vs)
u = np.linspace(-um, um, int(um/us/2)*2)
v = np.linspace(0, vm, int(vm/vs/2)*2)
uneg = u[u < 0]  # -u[1:][::-1]
u_non_neg = u[u >= 0]
# uc = np.concatenate([uneg, u])

sp = 8.0                # pixel size (microns)
lambd = 0.782 / sp      # wavelength
kp = 1.0                # transversal part of wave vector, to be re-escalated later as kp_pxsi
a_param = 5.0          # parameter "a" for Weber beam

# --- Confluent Hypergeometric Functions ---
zu = 1j * kp * u**2
zu_noneg = 1j * kp * u_non_neg**2
zv = 1j * kp * v**2

params = dict(a=a_param, kp=kp, readme=readme, lambd=lambd,
              sp=sp, um=um, vm=vm, us=us, vs=vs)

# Iterate through different parameters and collect results

# @title Iterate through different parameters and collect results
print("Iterating through new parameters to collect more hypergeometric results...")

# Increase mpmath precision to avoid potential numerical issues
mpmath.mp.dps = 50

# kp is set to 1, since only makes a reescalation of the coords
# Define new sets of parameters to test
new_params = [
    {'a_param': 0.0, 'kp': 1.0},
    {'a_param': 1.0, 'kp': 1.0},
    {'a_param': 5.0, 'kp': 1.0},
    {'a_param': 10.0, 'kp': 1.0},
    {'a_param': 20.0, 'kp': 1.0},
    {'a_param': 40.0, 'kp': 1.0},
]
hypergeometric_results = []  # Initialize list to store results
for params in new_params:
    a_param = params['a_param']
    kp = params['kp']
    print(f"\nCalculating for a_param={a_param}, kp={kp}...")

    # Recalculate dependent variables based on new a_param and kp
    # Ensure u, v are still in scope from previous cells
    # If not, you might need to re-run R83-Vs_JJ8or to get u, v
    zu = 1j * kp * u**2
    zu_noneg = 1j * kp * u_non_neg**2
    zv = 1j * kp * v**2

    # Recalculate hypergeometric functions
    start_time = time.perf_counter()
    F11Ue_nn = hyp1f1_vec(0.25 - 0.5j * a_param, 0.5, zu_noneg).astype(complex)
    F11Uo_nn = hyp1f1_vec(0.75 - 0.5j * a_param, 1.5, zu_noneg).astype(complex)
    end_time = time.perf_counter()
    print(f"U components calculation time: {end_time-start_time:.2f}s")

    start_time = time.perf_counter()
    F11Ve = hyp1f1_vec(0.25 + 0.5j * a_param, 0.5, zv).astype(complex)
    F11Vo = hyp1f1_vec(.75 + 0.5j * a_param, 1.5, zv).astype(complex)
    end_time = time.perf_counter()
    print(f"V components calculation time: {end_time-start_time:.2f}s")

    # Store the new results with their parameters
    hypergeometric_results.append({
        'a_param': a_param,
        'kp': kp,
        'F11Ue_nn': F11Ue_nn.copy(),
        'F11Uo_nn': F11Uo_nn.copy(),
        'F11Ve': F11Ve.copy(),
        'F11Vo': F11Vo.copy()
    })

print("Finished collecting results for all new parameters.")
print(f"Total sets of results collected: {len(hypergeometric_results)}")

# Displaying a summary of collected results
for i, result_set in enumerate(hypergeometric_results):
    print(f"\nResult set {i+1}:")
    print(f"  a_param: {result_set['a_param']}")
    print(f"  kp: {result_set['kp']}")
    print(f"  F11Ue_nn shape: {result_set['F11Ue_nn'].shape}")
    print(f"  F11Ve shape: {result_set['F11Ve'].shape}")

# timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # time stamp for saving
# %% save data

paramsf = dict(kp=kp, sp=sp, um=um, vm=vm, us=us, vs=vs, readme=readme)


def setup_output_directories(output_base: Path) -> dict:
    """Create output directories and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dirs = {
        'base': output_base,
        # 'figures': output_base / 'figures' / f'{timestamp}',
        'data': output_base / 'output_tests' / f'UV_HyperGF{timestamp}',
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


data_path = Path(r"data")
output_base = Path(PROJECT_ROOT / data_path)

dirs = setup_output_directories(output_base)

# subdir = f'free_fall_{timestamp}'

data_to_save = [{'common_data': paramsf,
                'a_hypergf': hypergeometric_results}]


filename = "hyperg_test_a0,5,10,20,40,us_0p1.pkl"
filePath = dirs['data'] / filename

# note "wb" for binary write
with open(filePath, "wb") as file:
    pickle.dump(data_to_save, file)

print(f"Data saved tp: {filePath}")
