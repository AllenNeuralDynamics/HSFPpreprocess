# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:41:27 2025

@author: carrie.stine
"""

#%% DESCRIPTION
# Use this notebook to execute the .py script while developing in the cloud

# Performs wavelength calibration to camera pixels.
# Uses the prism Sellmeier equation for the material N-SF11 to fit wavelength 
# to pixels.

# Inputs:
    # CalibrationImage.tiff: from pre-process set 1
# Outputs:
    # pixel_to_nm.hdf5: a 300x2 matrix in .hdf5 format with pixels as one 
    # column and wavelength that corresponds to that pixel as a second column.
    
#%% setup/imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
import argparse
from datetime import datetime as dt
import h5py
import ast
import CAS_preprocess_02_fibercoupled as pixel_to_nm

#%% variables
# Lasers used for calibration V1: original HSFP free-space rig
lasers = np.array([0.561, 0.514, 0.473, 0.445, 0.405])

# Lasers used for calibration V2: modified HSFP fiber-coupled rig 561
#lasers = np.array([0.561, 0.514, 0.488, 0.445, 0.405])

# Lasers used for calibration V3: modified HSFP fiber-coupled rig 594
#lasers = np.array([0.594, 0.514, 0.488, 0.445, 0.405])

session_id = "HSFP_775510_2025-02-20_11-08-27" # NEED TO CORRECT FOR CODE OCEAN

# Physical constants
# SELLMEIER_COEFFS = (1.73759695, 0.313747356, 1.89878101)
# SELLMEIER_TERMS = (0.013188707, 0.0623068142, 155.23629)
# THETA_I = 60.8  # angle of incidence (degrees)
# ALPHA = 60      # prism apex angle (degrees)
SAT_VAL = 7000  # value of laser saturation
DISTANCE = 50

#%% load calibration image
print("Starting HSFP image calibration processing step 2...")

# Settings   
data_dir =Path("C:/output_data") # NEED TO CORRECT FOR CODE OCEAN
# path = os.path.join(data_dir, session_id)
path = data_dir / session_id
results_path = os.path.join(path, 'fib') # NEED TO CORRECT FOR CODE OCEAN
# results_dir = Path(results_path)
results_dir = data_dir / session_id / 'fib' # NEED TO CORRECT FOR CODE OCEAN
             
             
calib_image = results_dir / 'CalibrationImage.tiff'
calib_file = results_dir / 'calibration.txt'

# find pixels corresponding to center position of each laser
laser_pix = pixel_to_nm.find_laser_pixels(calib_image, height_thresh=4000, dist_thresh=DISTANCE, sat_val=SAT_VAL)

#%% optimization function (return pixel value given angle of deviation as input)
popt, theta_D = pixel_to_nm.fit_wavelength_to_pixels(lasers, laser_pix)
print(f"Fit parameters: a={popt[0]:.4f}, b={popt[1]:.4f}")

wavelength = np.arange(0.4, 0.7, 0.001)
pixel_value = pixel_to_nm.linear_wave(pixel_to_nm.deviation_angle(wavelength), *popt)
pixel_to_nm.plot_calibration_results(lasers, laser_pix, wavelength, pixel_value)

# %% generate and save lut as hdf5 (pixel to nm table)
lut, pixel_value_wave, theta_D_wave = pixel_to_nm.generate_lut(popt, calib_file)
print(lut.head())

hdf5_file = results_dir / 'pixel_to_nm.hdf5'
store = pd.HDFStore(hdf5_file, mode='a')
for col in lut.columns:
    lut[col].to_hdf(store, key=col, mode='a')
store.close()

with h5py.File(hdf5_file, 'r') as f:
    print("HDF5 keys:", list(f.keys()))
    
print("Calibration processing step 2 complete.")

# %% plot angle of deviation
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(111, label='1')
ax1.plot(pixel_value_wave, 1000*wavelength, color='C0', linewidth=3)
ax1.set_xlabel('Camera pixel', color='C0')
ax1.set_ylabel('Wavelength (nm)', color='C3')
ax2 = fig.add_subplot(111, label='2', frame_on=False)
ax2.plot(theta_D_wave, 1000*wavelength, '--', color='C1', linewidth=2)
ax2.xaxis.tick_top()
ax2.set_xlabel('Angle of deviation', color='C1')
ax2.xaxis.set_label_position('top') 
plt.show()