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
    temp_fiber1 = np.zeros((frames_to_process, width), dtype=np.float64)
    temp_fiber2 = np.zeros((frames_to_process, width), dtype=np.float64)
    temp_peaks = np.zeros(frames_to_process, dtype=np.float64)

    current_frame = 0
    print(f"\nSegment {i}: Processing {frames_to_process} frames for {folder_path.name}...")

    # 2. Iterate through TIFF files in the folder
    for tiff_path in tiff_files:
        if current_frame >= frames_to_process:
            break
            
        # Load the entire stack (multi-page TIFF)
        stack = io.imread(tiff_path)
        # Ensure stack is 3D (may be 2D if it only contains 1 frame)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        # Define region used for background subtraction
        background = np.mean(stack[:,0:20,0:200]) # top left corner pixels used for bg subtract
        
        print(f"           Starting stack {tiff_path.name}")
        for f_idx in range(stack.shape[0]):
            if current_frame >= frames_to_process:
                break
            
            # Extract current frame and convert to float32 for math precision
            frame = stack[f_idx].astype(np.float32)
            
            
            # 3. Apply Transformations
            # M1 = Rotation, M2 = Affine Alignment
            frame_rotated = cv.warpAffine(frame, M1, (width, height))
            frame_aligned = cv.warpAffine(frame_rotated, M2, (width, height))
            # subtract background (if desired, Smrithi had this commented out)
            frame_aligned = frame_aligned #- background
            
            # 4. Extract Fiber Regions
            # identify the upper and lower bounds of each fiber
            f1_zone = frame_aligned[fiber1_location[1]:fiber1_location[0], :]
            f2_zone = frame_aligned[fiber2_location[1]:fiber2_location[0], :]
            
            # Take the mean across the Y-axis (height of the fiber) to get a 1D profile
            f1_mean = np.mean(f1_zone, axis=0)
            f2_mean = np.mean(f2_zone, axis=0)

            # 5. Peak Detection (using fiber 2)
            p_idx, _ = find_peaks(f2_mean, height=200, distance=200) 
            best_peak = 0
            if len(p_idx) > 0:
                best_peak = p_idx[np.argmax(f2_mean[p_idx])]
            
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


#%%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(np.array(frame), aspect='auto', vmin=0, vmax=6000)
axes[0].set(xlabel='Camera pixels', ylabel='Camera pixels', title='Raw Frame')
axes[0].grid(False)

im1 = axes[1].imshow(np.array(frame_rotated), aspect='auto', vmin=0, vmax=6000)
axes[1].set(xlabel='Camera pixels', ylabel='Camera pixels', title='Rotated Frame')
axes[1].grid(False)

im2 = axes[2].imshow(np.array(frame_aligned), aspect='auto', vmin=0, vmax=6000)
axes[2].set(xlabel='Camera pixels', ylabel='Camera pixels', title='Final Transformed Frame')
axes[2].grid(False)

plt.show()


#%% 
# numFrames = np.zeros(len(metadata_files), dtype=int) # store the number of frames in each metadata file
# time = [] # initialize to store camera timestamps
# Frames = [] # initialize to store corrected framestamps after handling rollovers

# # Helper: build unified timestamps from available columns
# def build_timestamps(df: pd.DataFrame) -> np.ndarray:
#     cols = df.columns
#     if {"CameraTimestampSeconds", "CameraTimestampMicroSeconds"}.issubset(cols):
#         ts = df["CameraTimestampSeconds"].astype("float64") \
#              + df["CameraTimestampMicroSeconds"].astype("float64") * 1e-6
#     elif "CameraTimestamp" in cols:
#         ts = df["CameraTimestamp"].astype("float64")
#     else:
#         raise KeyError("Metadata missing timestamp columns: expected either "
#                        "('CameraTimestampSeconds' + 'CameraTimestampMicroSeconds') or 'CameraTimestamp'.")
#     return ts.to_numpy()

# # Helper: correct 16-bit framestamp rollover
# def correct_framestamps(df: pd.DataFrame) -> np.ndarray:    # input = pandas df, output = numpy array
#     if "Framestamp" not in df.columns:
#         raise KeyError("Metadata missing 'Framestamp' column.")

#     # Use uint32 to store raw counter safely, then int64 for corrected indices
#     fs = df["Framestamp"].to_numpy(dtype=np.uint32)
#     fs_signed = fs.astype(np.int64) # convert to signed type before calculating diffs
#     # Detect rollovers: when the counter decreases from one frame to the next
#     # Example: [..., 65535, 0, 1, ...] -> np.diff < 0 at the rollover boundary
#     diffs = np.diff(fs_signed)
#     rollover_points = np.r_[False, diffs < 0]           # prepend False for the first frame
#     rollover_count = np.cumsum(rollover_points).astype(np.int64)

#     # Each rollover adds 65536 to subsequent frames
#     fs_corrected = fs.astype(np.int64) + rollover_count * 65536

#     return fs_corrected

# # Main loop over metadata files
# for i, meta_path in enumerate(metadata_files):
#     # pandas can read Path objects directly
#     metadata = pd.read_csv(meta_path)

#     # Count frames (rows) robustly
#     numFrames[i] = int(metadata.shape[0])

#     # Build timestamps and corrected framestamps
#     ts = build_timestamps(metadata)
#     fs_corr = correct_framestamps(metadata)

#     time.append(ts)
#     Frames.append(fs_corr)

# # (Optional) Sanity checks/diagnostics
# print(f"Read {len(metadata_files)} metadata file(s).")
# print("Frames per file:", numFrames.tolist())

# # Example: show detected rollovers per file
# rollovers_per_file = []

# for i, meta_path in enumerate(metadata_files):
#     md = pd.read_csv(meta_path)
#     fs_i = md["Framestamp"].astype(np.int64).to_numpy()
#     diffs = np.diff(fs_i)
#     rollover_points = diffs < 0
#     rollovers_per_file.append(int(rollover_points.sum()))  # number of True values
# print("Detected rollovers per file:", rollovers_per_file)
























# %% Get all Tiff directories
files = os.listdir(results_dir)
print(files)
folders = []

# Put all directories containing 'Tiff' into 'folders'
for entry in os.scandir(results_dir):
    if entry.is_dir() and 'Tiff' in entry.name:
        folders.append(entry.name)
folders = natsorted(folders)
print(folders)

# %% # --- Initialize storage lists ---
fiber1 = []
fiber2 = []
peaks = []

tiffbatch = 1000  # number of frames per Tiff file

# Loop over each folder containing Tiff stacks
for folder_name in folders:
    folder_path = os.path.join(results_dir, folder_name)
    tiff_files = natsorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])
    
    num_files = len(tiff_files)
    # Note: need to figure out how to index current time array
    #lenToUse = min(num_files, math.floor(len(time[i])/1000))  # number of Tiffs to process; adjust as needed
    
    print(f"Processing folder: {folder_name}")
    print(f"  Total TIFFs detected: {num_files}")
    print(f"  TIFFs that will be processed: {lenToUse}")

    # Pre-allocate arrays
    sz_x = tiffbatch * lenToUse  # total frames (frames per tiff x number of tiffs)
    sz_y = width  # width of each frame
    tempfiber1 = np.zeros((sz_x, sz_y)) # will hold fiber1 values for each tiff
    tempfiber2 = np.zeros((sz_x, sz_y)) # will hold fiber2 values for each tiff
    peak_array = np.zeros(sz_x) # will hold detected peak positions for each tiff

    # Loop over selected Tiff files in the current Tiff folder
    for t_idx, tiff_name in enumerate(tiff_files[:lenToUse]):
        tiff_path = os.path.join(folder_path, tiff_name) # create path to current tiff stack
        tiff_stack = io.imread(tiff_path).astype(float) # read in the current tiff stack
        num_frames = tiff_stack.shape[0]

        for f_idx in range(num_frames):
            # Apply rotation 
            temp_rotated = cv.warpAffine(tiff_stack[f_idx, :, :], M1, (cols, rows))
            # Apply affine transformation
            img = cv.warpAffine(temp_rotated, M2, (cols, rows))
            
            # Extract fiber regions
            fiber1_m = np.mean(img[fiber1_location[1]:fiber1_location[0], :], axis=0)
            fiber2_m = np.mean(img[fiber2_location[1]:fiber2_location[0], :], axis=0)

            # Detect peaks for fiber2
            temp_peak, _ = find_peaks(fiber2_m, height=200, distance=200) 
            if len(temp_peak) > 1:
                max_idx = np.argmax(fiber2_m[temp_peak])
                temp_peak = temp_peak[max_idx]
            # elif len(temp_peak) == 0:
            #     temp_peak = 0  # default if no peak found

            # Store results
            frame_index = t_idx * tiffbatch + f_idx
            tempfiber1[frame_index, :] = fiber1_m
            tempfiber2[frame_index, :] = fiber2_m
            peak_array[frame_index] = temp_peak

    # Append processed data for this folder
    fiber1.append(tempfiber1)
    fiber2.append(tempfiber2)
    peaks.append(peak_array)

# Truncate time arrays to match fiber frame counts
for i in range(len(time)):
    if len(time[i]) > fiber1[i].shape[0]:
        time[i] = time[i][:fiber1[i].shape[0]]


