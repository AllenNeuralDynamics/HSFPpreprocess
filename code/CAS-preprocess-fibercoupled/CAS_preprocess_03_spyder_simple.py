# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:54:07 2025

@author: carrie.stine
"""

# %% DESCRIPTION
# Pre-process step 3
# - Converts the raw tiff images acquired by the camera to a single intensity
#  value for each fiber at each wavelength. 
# - Needs:
    # - Session data csv file saved by the Hamamatsu Image Processing code. 
        # Contains metadata such as frame number, ROI dimensions in the camera 
        # sensor, timestamps, etc.
    # - calibration.txt file that contains information to transform the images 
        # before intensity extraction.
    # - Raw data saved as tiff file stacks of 1000 frames per stack.
# - Steps:
    # 1. Extract each tiff frame.
    # 2. Rotate and affine transform the image.
    # 3. Extract intensity for each fiber and average the intensity for all 
        # pixels in y-direction.
    # 4. Find the laser sequence and deinterleave the singals. 
    # 5. Convert pixel data to wavelength.
    # 6. Low pass filter the signal.
# - Saves:
    # - Timestamp vectors, wavelength vector, intensity over time matrix for 
        # each laser wavelength.
    # - Saved in hdf5 format.

#%% QUESTIONS
# lenToUse? Note says added for ocamp testing
# not using top left pixels for background subtraction anymore?

# %% setup/imports
import pandas as pd
import numpy as np
import math
import os
import glob
from natsort import natsorted
from skimage import io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import itertools
import cv2 as cv
import h5py
import ast
from PIL import Image
from pathlib import Path
import logging
import tifffile
import CAS_preprocess_03_fibercoupled as tiff_to_intensity

# %% variables
# store the session id
session_id = 'HSFP_775510_2025-02-20_11-08-27'

# %% # path settings  
data_dir =Path("C:/output_data") # NEED TO CORRECT FOR CODE OCEAN
path = data_dir / session_id
results_dir = data_dir / session_id / 'fib' # NEED TO CORRECT FOR CODE OCEAN
             
calib_image = results_dir / 'CalibrationImage.tiff'
calib_file = results_dir / 'calibration.txt'

if not results_dir.exists():
    raise FileNotFoundError(f"Results directory not found: {results_dir}")
if not calib_file.exists():
    raise FileNotFoundError(f"Calibration file not found: {calib_file}")


#%% load session_params.csv and fix framestamp rollover
# NOTE: load_session_params drops the last frame in the metadata file since this is often corrupt!
    # num_frames = array of size x where x is the number of recordings in the folder 
            # values are # of frames in a given recording
    # times_raw = list of size x where x is the number of recordings in the folder
            # each entry is a time series array (using camera timestamps) the length of corresponding num_frames
    # frames_raw = list of size x where x is the number of recordings in the folder
            # each entry is an array of frame counts the length of corresponding num_frames (use this to ID frame drops)
    # metadata_files = list of size x where x is the number of recordings in the folder  
            # each entry is a path to the metadata file for the corresponding recording
num_frames, times_raw, frames_raw, metadata_files = tiff_to_intensity.load_session_params(results_dir)

# identify the session param files that were found in the folder, there should be 1 per recording
print("Processed CSV files (in order):")
for p in metadata_files:
    print("  -", p.name)

print("\nnumber of frames per recording fragment:", num_frames)
for i, (t_arr, f_arr) in enumerate(zip(times_raw, frames_raw)):
    print(f"\nRecording {i}:")
    #print(f"  time: array length = {t_arr.size}, example [-5:] = {t_arr[-5:]}")
    print(f"  Frames (rollover-corrected): length = {f_arr.size}, example [-5:] = {f_arr[-5:]}")


#%% check for dropped frames and repair by interpolating time stamps
# frames = list where each entry contains frame counts for each recording after drop correction
# times = list where each entry contains camera timestamps for each recording, interpolated after drop correction
frames, times = tiff_to_intensity.check_frame_drop(frames_raw, times_raw)

for i, (t_arr, f_arr) in enumerate(zip(times, frames)):
    print(f"\nRecording {i}:")
    #print(f"  time: array length = {t_arr.size}, example [-5:] = {t_arr[-5:]}")
    print(f"  Frames (drop-corrected): length = {f_arr.size}, example [-5:] = {f_arr[-5:]}")    
    
    

#%% read in camera dimensions and offsets, convert to integers
metadata = pd.read_csv(metadata_files[0])
width = metadata.Width[0]
height = metadata.Height[0]
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

print('\nWidth: ' + str(width))
print('Height: ' + str(height))
print('X Offset: ' + str(Xoffset))
print('Y Offset: ' + str(Yoffset))

#%% extract values from calibration.txt
theta_r, pt1, pt2, pt3, pt4, pt5, pt6, fiber1_location, fiber2_location = tiff_to_intensity.load_calib_values(calib_file, Xoffset, Yoffset)


#%% Create rotation and affine transformation matrix for the recording
rows,cols = [height, width]
M1 = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),theta_r,1)
pts1 = np.float32([pt1, pt2, pt3])
pts2 = np.float32([pt4, pt5, pt6])
M2 = cv.getAffineTransform(pts1,pts2)

# pre-compute the combined matrix transformation
M1_3x3 = np.vstack([M1, [0,0,1]])
M2_3x3 = np.vstack([M2, [0,0,1]])
M_combined = np.dot(M2_3x3, M1_3x3)[:2, :] # result is a 2x3
#%% check tiff count vs metadata frame count for alignment concerns
# Configure logging to track alignment issues
logging.basicConfig(level=logging.INFO)

# initialize storage lists
meta_frame_counts = []
tiff_frame_counts = []

# get the tiff directories
files = os.listdir(results_dir)
print('\nFiles in path:')
print(files)

# Put all directories containing 'Tiff' into 'folders'
folders = natsorted([f for f in results_dir.iterdir() if f.is_dir() and f.name.startswith('Tiffs')])
print('\nTiff folders:')
print(folders)

#%%
# loop over all tiff folders:
for i, folder_path in enumerate(folders):
    # 1. Gather all TIFFs in this recording fragment
    tiff_files = natsorted(list(folder_path.glob('*.tif*')))
    total_tiffs = 0

    # 2. Calculate total number of images per recording
    for tiff_p in tiff_files:
        with Image.open(tiff_p) as img:
            total_tiffs += img.n_frames
    
    # Store image counts using append, DROP FINAL IMAGE FROM COUNT to match metadata drop (last frame often corrupt)
    tiff_frame_counts.append(total_tiffs - 1) 
    meta_frame_counts.append(len(frames[i]))
            
   # Print diagnostic for this segment
    print(f"\n Segment {i} ({folder_path.name}):")
    print(f"  Metadata: {meta_frame_counts[i]} frames")
    print(f"  TIFFs:    {tiff_frame_counts[i]} frames")
    
    if tiff_frame_counts[i] > meta_frame_counts[i]:
        print(f"WARNING: Frame count mismatch detected! More tiffs than metadata frames; trailing TIFFs will be ignored. Using minimum: {min(tiff_frame_counts[i], meta_frame_counts[i])}")

    if tiff_frame_counts[i] < meta_frame_counts[i]:
        raise ValueError(
            f"\n Alignment Error in Segment {i}:\n"
            f"Metadata ({meta_frame_counts} frames) exceeds TIFF count ({tiff_frame_counts} frames).\n"
            f"This suggests images were lost or the video writer crashed early.\n"
            f"Manual inspection required for {folder_path.name}."
        )


#%% extract data from tiff stacks

# initialize storage lists to contain n x m arrays where n = # of frames and m = width of image
fiber1 = [] # 1 entry/segment with mean intensity avgd across height of fiber 1
fiber2 = [] # 1 entry/segment with mean intensity avgd across height of fiber 2
peaks = [] # 1 entry/segment with location of peaks for each frame (peaks identified using fiber 2)

for i, folder_path in enumerate(folders):
    # Calculate how many frames we can safely process
    frames_to_process = min(meta_frame_counts[i], tiff_frame_counts[i])
    tiff_files = natsorted(list(folder_path.glob('*.tif*')))

    # 1. Pre-allocate NumPy arrays for this segment
    # Shape: (Frames, Width of the Fiber)
    temp_fiber1 = np.zeros((frames_to_process, width), dtype=np.float32) # 1/27/26 - changing these from float64 to float 32
    temp_fiber2 = np.zeros((frames_to_process, width), dtype=np.float32)
    temp_peaks = np.zeros(frames_to_process, dtype=np.float32)

    current_frame = 0
    print(f"\nSegment {i}: Processing {frames_to_process} frames for {folder_path.name}...")

    # 2. Iterate through TIFF files in the folder
    for tiff_path in tiff_files:
        if current_frame >= frames_to_process:
            break
            
        print(f"           Starting stack {tiff_path.name}")
        
        # NEW 1/27/26 uses tifffile to 'stream' ind frames and prevent crash from loading full image at once
        with tifffile.TiffFile(tiff_path) as tif:
            # Calculate background once per stack using the first frame
            first_frame = tif.pages[0].asarray().astype(np.float32)
            background = np.mean(first_frame[0:20, 0:200])
            
            for page in tif.pages:
                if current_frame >= frames_to_process:
                    break
                
                # stream the current frame
                frame = page.asarray().astype(np.float32)
                
                # OPTIONAL: subtract background (if desired, Smrithi had this commented out)
                # frame = frame - background
                
                # single-step transformation
                frame_aligned = cv.warpAffine(frame, M_combined, (width, height))
                
                
                # 4. Extract Fiber Regions
                # identify the upper and lower bounds of each fiber
                # NOTE: upper bound [1] is the smaller number, must go from smaller to larger in slicing
                f1_zone = frame_aligned[fiber1_location[1]:fiber1_location[0], :]
                f2_zone = frame_aligned[fiber2_location[1]:fiber2_location[0], :]
            
                # Take the mean across the Y-axis (height of the fiber) to get a 1D profile
                f1_mean = np.mean(f1_zone, axis=0)
                f2_mean = np.mean(f2_zone, axis=0)

                # 5. Peak Detection (using fiber 2)
                p_idx, _ = find_peaks(f2_mean, height=200, distance=200) 
                best_peak = p_idx[np.argmax(f2_mean[p_idx])] if len(p_idx) > 0 else 0
            
                # 6. Assign to pre-allocated arrays
                temp_fiber1[current_frame, :] = f1_mean
                temp_fiber2[current_frame, :] = f2_mean
                temp_peaks[current_frame] = best_peak
            
                current_frame += 1

    # Append the completed segment arrays to our master lists
    fiber1.append(temp_fiber1)
    fiber2.append(temp_fiber2)
    peaks.append(temp_peaks)

    # 7. Final Sync: Trim metadata to match actual processed frame count
    times[i] = times[i][:frames_to_process]
    frames[i] = frames[i][:frames_to_process]

print("\nData extraction complete.")


#%% Plot an example frame before + after affine transformation
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(np.array(frame), aspect='auto', vmin=0, vmax=6000)
axes[0].set(xlabel='Camera pixels', ylabel='Camera pixels', title='Raw Frame')
axes[0].grid(False)

# im1 = axes[1].imshow(np.array(frame_rotated), aspect='auto', vmin=0, vmax=6000)
# axes[1].set(xlabel='Camera pixels', ylabel='Camera pixels', title='Rotated Frame')
# axes[1].grid(False)

im1 = axes[1].imshow(np.array(frame_aligned), aspect='auto', vmin=0, vmax=6000)
axes[1].set(xlabel='Camera pixels', ylabel='Camera pixels', title='Final Transformed Frame')
axes[1].grid(False)

plt.show()

#%% Plot fiber bounds to check ROI alignment
# 1. Create a mean image from a subset of frames (or use the calibration image)
# We use the first segment 'temp_fiber1' as a reference
sample_frame = frame_aligned.copy() # Use the last processed frame from your loop

plt.figure(figsize=(12, 6))
plt.imshow(sample_frame, cmap='gray', aspect='auto')

# 2. Draw the ROI boundaries for Fiber 1
plt.axhline(y=fiber1_location[0], color='r', linestyle='--', alpha=0.8, label='Fiber 1 Bounds')
plt.axhline(y=fiber1_location[1], color='r', linestyle='--', alpha=0.8)

# 3. Draw the ROI boundaries for Fiber 2
plt.axhline(y=fiber2_location[0], color='cyan', linestyle='--', alpha=0.8, label='Fiber 2 Bounds')
plt.axhline(y=fiber2_location[1], color='cyan', linestyle='--', alpha=0.8)

plt.title(f"ROI Alignment Check: {session_id}")
plt.xlabel("Wavelength / Pixels (Width)")
plt.ylabel("Vertical Position (Height)")
plt.legend(loc='upper right')

# Zoom in to the fiber areas to see the tilt better
plt.ylim(max(fiber2_location)+20, min(fiber1_location)-20) 
plt.show()





#%% Set order of lasers and interleave into individual laser channels

lut_pix = pd.read_hdf(results_dir/'pixel_to_nm.hdf5', key='Camera_pixel', more='r')
lut_wav = pd.read_hdf(results_dir/'pixel_to_nm.hdf5', key='Wavelength_nm', more='r')
wavelength = w.to_numpy()
camera_px = c.to_numpy()
lasers = [405,445,473,514,561]

laser_order = []
for i in range(len(peaks)):
    l_order = np.zeros(len(peaks[i]))
    for j in range(len(peaks[i])):
        laser_pix = min(camera_px, key=lambda x:abs(x-peaks[i][j]-Xoffset))
        camera_pix = np.where(camera_px>=laser_pix)
        p = camera_pix[-1]
        p = p[-1]
        temp_laser = min(lasers, key=lambda x:abs(x-wavelength[p]))
        l_order[j] = temp_laser
    laser_order.append(l_order)
print(laser_order)

# Plot the peaks to check if the laser order is correct
for i in range(len(peaks)):
    plt.plot(peaks[i],'.')
plt.show()
