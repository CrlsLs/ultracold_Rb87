# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 22:57:54 2026

@author: clher
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import k as k_B, hbar, m_u
from scipy.optimize import curve_fit
from scipy.stats import norm
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import sys
import scipy.io
import numpy as np

mat_data = scipy.io.loadmat('Psi_a10_kp3.mat')


# PROJECT_ROOT = Path(__file__).parent.parent
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

# # Define the path using pathlib
# # It automatically formats the slashes correctly for your OS
# folder_path = Path("C:\Users\clher\Documents\programacion\projects\ultracold_Rb87\data")

# file_name = "my_data.mat"

# # Combine them using the '/' operator (pathlib allows this!)
# full_path = folder_path / file_name

# # Load the file using the complete path
# mat_data = scipy.io.loadmat(full_path)

# # Load the .mat file into a dictionary
# mat_data = scipy.io.loadmat('Psi_a10_kp3.mat')

# # To see what variables are stored in the file, you can print the keys:
print(mat_data.keys())

# # Extract your specific matrix using its MATLAB variable name
# # (Replace 'my_matrix_name' with the actual name from your file)
my_matrix = mat_data['Psi']

# # Verify it is a NumPy array
print(type(my_matrix))
print(my_matrix.shape)

# %%

data = sio.loadmat('Psi_a10_kp3.mat')
print(data.keys())  # see variables inside the file
# %%
A = data['Psi']   # if the matrix variable in MATLAB was called A
