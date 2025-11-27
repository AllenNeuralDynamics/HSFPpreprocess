# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:29:03 2025

@author: svc_aind_behavior
"""
#%% Setup/imports
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
SAT_VAL = 12000 # saturation value of camera
FIBER_WIDTH = 40 # width of fiber (in pixels)

# store the session id
session_id = "815738_2025-11-25T11_19_02.6493184-08_00"

#%% main
print("Starting HSFP image calibration processing...")
        
# Settings
data_dir = r"C:\output_data\\"

# Get session ID
path, calib_path = unskew_image.load_session_paths(data_dir, session_id)
metadata = unskew_image.load_calibration_metadata(calib_path)
tiff_dir = os.path.join(calib_path, 'Tiffs')
img2d = unskew_image.load_and_average_tiff(tiff_dir)

# Camera offsets
Xoffset = int(metadata.XOffset[0])
Yoffset = int(metadata.YOffset[0])

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


# Analyze fibers
v_width1, h_width1 = unskew_image.analyze_laser(img_rotated, h_peaks_rot, use_laser=0)
v_width2, h_width2 = unskew_image.analyze_laser(img_rotated, h_peaks_rot, use_laser=2)

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

fiber1, fiber2 = unskew_image.store_fiber_boundaries(img_final, h_peaks_final, use_laser=2)

# Save results
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

print("Calibration processing complete.")