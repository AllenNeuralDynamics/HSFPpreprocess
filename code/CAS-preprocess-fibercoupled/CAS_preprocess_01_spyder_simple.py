# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:29:03 2025

@author: svc_aind_behavior
"""

#%% DESCRIPTION
# Use this notebook to execute the equivalent .py script while developing in 
# the cloud.

# This pre processing step first rotates the full frame to flatten the spectrum
# using rotations transformation.
# Following rotation, the image is affine transformed to straighten the skew in 
# the slit due to the prism.
# Acquire a calibration sequence after any modifications to the system or at
# the start of every month if no modifications are made.
    # Turn on all lasers and acquire a short sequence at full frame.
    
# Inputs:
    # Calibration sequence (folder in data/session_id/fib/ called CalibrationFiles) containing:
        # session_params.csv: csv containing camera parameters for width, height, Xoffset, and Yoffset
        # Tiffs: a subfolder with tiffs named Tiffs0.tif, Tiffs1.tif, Tiffs2.tif, etc
# Outputs:
    # CalibrationImage.tiff: a .tiff file with the final calibrated image.
    # calibration.txt: a .txt file with the transformation points.

#%% setup/imports
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from scipy.signal import find_peaks
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import CAS_preprocess_01_fibercoupled as unskew_image

warnings.filterwarnings("ignore")

#%% variables
SAT_VAL = 7000 # saturation value of camera
FIBER_WIDTH = 40 # width of fiber (in pixels)
USE_LASER_1 = 0 # first laser used for calculating affine transformation 
USE_LASER_2 = 2 # second laser used for calculating affine transformation

# store the session id
session_id = "836733_2025-12-03T10_34_41.0261632-08_00"

#%% load calibration image
print("Starting HSFP image calibration processing step 1...")
        
# Settings
data_dir = r"C:\output_data\\"

# Get session ID
path, calib_path = unskew_image.load_session_paths(data_dir, session_id)
metadata = unskew_image.load_calibration_metadata(calib_path)
tiff_dir = os.path.join(calib_path, 'Tiffs')
img2d = unskew_image.load_and_average_tiff(tiff_dir)

# Camera offsets
# Use XOffset if available (Bonsai node V3.2) or Left if not (V4)
if hasattr(metadata, "XOffset"):
    Xoffset = int(metadata.XOffset[0])
else:
    Xoffset = int(metadata.Left[0])

# Use 'YOffset' if available, otherwise fall back to 'Top'
if hasattr(metadata, "YOffset"):
    Yoffset = int(metadata.YOffset[0])
else:
    Yoffset = int(metadata.Top[0])

#%% plot averaged image
f,ax = plt.subplots(figsize=(6,6))
i = ax.imshow( np.array(img2d), aspect='auto', vmin=0, vmax=SAT_VAL)
ax.set(xlabel='Camera pixels', ylabel='Camera pixels', title='Calibration Image')
ax.grid(False)
f.colorbar(i,ax=ax)
plt.show()

#%% rotate the image
# Find laser positions
h_peaks, v_peaks, img_to_unskew = unskew_image.find_laser_positions(img2d)

# Rotate image
img_rotated, theta_r = unskew_image.rotate_image(img_to_unskew, h_peaks, v_peaks)

# Calculate centers of each laser after rotation
h_line = np.mean(img_rotated, axis=0)
h_peaks_rot, _ = find_peaks(h_line, height=2000, distance=50)

v_peaks_rot = np.zeros(np.size(h_peaks_rot))
for i, x in enumerate(h_peaks_rot):
    v_line = img_rotated[:, x].copy()
    v_line[v_line < SAT_VAL] = 0
    idx = np.nonzero(np.diff(v_line))[0]

    if len(idx) >= 2:  # Need at least 2 edges
        v_peaks_rot[i] = int(np.round((idx[0] + idx[-1]) / 2))
    else:
        print(f"Warning: Could not find edges for laser at x={x}")
v_peaks_rot = v_peaks_rot.astype(int)

#%% plot the rotated image
f,ax = plt.subplots(figsize=(6,6))
i = ax.imshow(img_rotated, aspect='auto', vmin=0, vmax=SAT_VAL)
ax.set(xlabel='Camera pixels', ylabel='Camera pixels', title='Rotated Image')
ax.grid(False)
f.colorbar(i,ax=ax)
plt.show()

#%% analyze fibers
v_width1, h_width1 = unskew_image.analyze_laser(img_rotated, h_peaks_rot, use_laser=USE_LASER_1)
v_width2, h_width2 = unskew_image.analyze_laser(img_rotated, h_peaks_rot, use_laser=USE_LASER_2)

# Define affine points and transform image
pt1, pt2, pt3 = [h_width1[0], v_width1[0]], [h_width1[1], v_width1[1]], [h_width2[0], v_width2[0]]
pt4, pt5, pt6 = [h_width1[1], v_width1[0]], [h_width1[1], v_width1[1]], [h_width2[1], v_width2[0]]

img_final = unskew_image.perform_affine_transform(img_rotated, [pt1, pt2, pt3], [pt4, pt5, pt6])


# Store fiber boundaries
h_line_final = np.mean(img_final, axis=0)
h_peaks_final, _ = find_peaks(h_line_final, height=2000, distance=50)

v_peaks_final = np.zeros(np.size(h_peaks_final))
for i, x in enumerate(h_peaks_final):
    v_line = img_final[:, x].copy()
    v_line[v_line < SAT_VAL] = 0
    idx = np.nonzero(np.diff(v_line))[0]

    if len(idx) >= 2:  # Need at least 2 edges
        v_peaks_final[i] = int(np.round((idx[0] + idx[-1]) / 2))
    else:
        print(f"Warning: Could not find edges for laser at x={x}")
v_peaks_final = v_peaks_final.astype(int)

fiber1, fiber2 = unskew_image.store_fiber_boundaries(img_final, h_peaks_final, use_laser=USE_LASER_2)

#%% save results
pt1[0] = int(pt1[0] + Xoffset)
pt2[0] = int(pt2[0] + Xoffset)
pt3[0] = int(pt3[0] + Xoffset)
pt4[0] = int(pt4[0] + Xoffset)
pt5[0] = int(pt5[0] + Xoffset)
pt6[0] = int(pt6[0] + Xoffset)
pt1[1] = int(pt1[1] + Yoffset)
pt2[1] = int(pt2[1] + Yoffset)
pt3[1] = int(pt3[1] + Yoffset)
pt4[1] = int(pt4[1] + Yoffset)
pt5[1] = int(pt5[1] + Yoffset)
pt6[1] = int(pt6[1] + Yoffset)
fiber1[0] = int(fiber1[0] + Yoffset)
fiber1[1] = int(fiber1[1] + Yoffset)
fiber2[0] = int(fiber2[0] + Yoffset)
fiber2[1] = int(fiber2[1] + Yoffset)

#create a path to the calibration files
results_path = os.path.join(path, 'fib')
results_dir = Path(results_path)
unskew_image.save_results(img_final, theta_r, [pt1, pt2, pt3, pt4, pt5, pt6], [fiber1, fiber2], Xoffset, Yoffset, results_dir)

print("Calibration processing step 1 complete.")

#%% create 1x3 subplots to show raw, rotated, and transformed image

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Left: Original image with midpoints ---
im0 = axes[0].imshow(img_to_unskew, aspect='auto', vmin=0, vmax=SAT_VAL)
axes[0].plot(h_peaks, v_peaks, 'ro', markersize=5, label='Laser midpoints')
axes[0].set(title='Calibration Image - Raw', xlabel='Camera pixels', ylabel='Camera pixels')
axes[0].legend()
#cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
#cbar0.set_label('Pixel Intensity')

# --- Middle: Rotated image with midpoints ---
im1 = axes[1].imshow(img_rotated, aspect='auto', vmin=0, vmax=SAT_VAL)
axes[1].plot(h_peaks_rot, v_peaks_rot, 'ro', markersize=5, label='Laser midpoints')
axes[1].set(title='Calibration Image - Rotated', xlabel='Camera pixels', 
            #ylabel='Camera pixels'
           )
axes[1].yaxis.set_visible(False)
axes[1].legend()
#cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04) # uncomment to give plot its own colorbar
#cbar1.set_label('Pixel Intensity')

# --- Right: Final image with midpoints ---
im2 = axes[2].imshow(img_final, aspect='auto', vmin=0, vmax=SAT_VAL)
axes[2].plot(h_peaks_final, v_peaks_final, 'ro', markersize=5, label='Laser midpoints')

# Add horizontal lines for fiber boundaries
axes[2].axhline((fiber1[0]-Yoffset), color='red', linestyle='--', linewidth=2, label='Fiber 1 Boundaries')
axes[2].axhline((fiber1[1]-Yoffset), color='red', linestyle='--', linewidth=2, 
                #label='Fiber1 Bottom'
               )
axes[2].axhline((fiber2[0]-Yoffset), color='blue', linestyle='--', linewidth=2, label='Fiber 2 Boundaries')
axes[2].axhline((fiber2[1]-Yoffset), color='blue', linestyle='--', linewidth=2, 
                #label='Fiber2 Bottom'
               )

axes[2].set(title='Calibration Image - Final', xlabel='Camera pixels', 
            #ylabel='Camera pixels'
           )
axes[2].yaxis.set_visible(False)
axes[2].legend(loc='upper right')

cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_label('Pixel Intensity', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()