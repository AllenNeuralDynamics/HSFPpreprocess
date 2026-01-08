# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:34:29 2025

@author: carrie.stine
"""
#%% setup/imports
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
from PIL import Image
from pathlib import Path
#%% correct bit rollover
# store the session id
session_id = 'HSFP_775510_2025-02-20_11-08-27'

data_dir =Path("C:/output_data") # NEED TO CORRECT FOR CODE OCEAN
path = data_dir / session_id
results_dir = data_dir / session_id / 'fib' # NEED TO CORRECT FOR CODE OCEAN
metadata_file = natsorted(list(results_dir.glob("session*.csv")))
print(metadata_file)

numFrames = np.zeros(len(metadata_file))
time = []
framestamp = []
Frames = []
for i in range(len(metadata_file)):
    metadata = pd.read_csv(metadata_file[i])
    numFrames[i] = len(metadata.Width)
    numFrames[i] = numFrames[i].astype(int)
    
    md_cols = metadata.columns
    if {"CameraTimestampSeconds", "CameraTimestampMicroSeconds"}.issubset(md_cols):
        timestamp = metadata.CameraTimestampSeconds + 10**-6*metadata.CameraTimestampMicroSeconds        
    elif "CameraTimestamp" in md_cols:
        timestamp = metadata.CameraTimestamp
    
    time.append(timestamp)    

    framestamp = metadata.Framestamp
    framestamp = np.array(framestamp)
    bitmax = np.where(framestamp == 65535)
    if len(bitmax[0]) == 1:
        cut = framestamp[bitmax[0][0]+1:-1]
        newcut = cut + 65535
        Frames.append(np.concatenate((framestamp[0:bitmax[0][0]],newcut)))
    elif len(bitmax[0]) > 1:
        temp = []
        temp.append(np.array(framestamp[0:bitmax[0][0]]))
        for j in range(len(bitmax[0])-1):
            cut = temp[j][-1] + framestamp[bitmax[0][j]+1:bitmax[0][j+1]]
            temp.append(cut)
        temp.append(temp[j+1][-1] + framestamp[bitmax[0][-1]+1:-1])
        Frames.append(np.concatenate(temp))
    elif len(bitmax[0]) == 0:
        Frames.append(framestamp[0:-1])

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

#%% # Unskew image, rotation followed by affine transformation

# Read from calibration.txt
with open(results_dir / 'calibration.txt', 'r') as f:
    for line in f:
        name, value = line.strip().split(' = ')
        exec(f'{name} = {value}')

# Read from above txt file after running preprocess_01_unskewimage
theta_r = rot_tform_thetaR
pt1 = [aff_tform_pt1[0]-Xoffset, aff_tform_pt1[1]-Yoffset] 
pt2 = [aff_tform_pt2[0]-Xoffset, aff_tform_pt2[1]-Yoffset] 
pt3 = [aff_tform_pt3[0]-Xoffset, aff_tform_pt3[1]-Yoffset] 
pt4 = [aff_tform_pt4[0]-Xoffset, aff_tform_pt4[1]-Yoffset] 
pt5 = [aff_tform_pt5[0]-Xoffset, aff_tform_pt5[1]-Yoffset] 
pt6 = [aff_tform_pt6[0]-Xoffset, aff_tform_pt6[1]-Yoffset]
fiber1_location = [fiber1_pixels[1]-Yoffset,fiber1_pixels[0]-Yoffset]
fiber2_location = [fiber2_pixels[1]-Yoffset,fiber2_pixels[0]-Yoffset]

fiber1_location = [int(x) for x in fiber1_location]
fiber2_location = [int(x) for x in fiber2_location]

# Create rotation and affine transformation matrix
rows,cols = [height, width]
M1 = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),theta_r,1)
pts1 = np.float32([pt1, pt2, pt3])
pts2 = np.float32([pt4, pt5, pt6])
M2 = cv.getAffineTransform(pts1,pts2)

# %% Get all Tiff directories
files = os.listdir(results_dir)
print(files)
folders = []

# def find_dirs_with_string(folder_path, search_string):
#     return [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and search_string in d]

# # Example usage
# folder_path = paths['raw_data']
# search_string = "Tiff"
# folders = find_dirs_with_string(folder_path, search_string)

for entry in os.scandir(results_dir):
    if entry.is_dir() and 'Tiff' in entry.name:
        folders.append(entry.name)
folders = natsorted(folders)
print(folders)

# %% Read through each Tiff directory, extract tiff files, and unskew images
fiber1 = []
fiber2 = []
peaks = []
tiffbatch = 1000
for i in range(len(folders)):
    print(folders[i])
    folder_path = os.path.join(results_dir, folders[i])
    files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
    #files = os.listdir(os.path.join(results_dir,folders[i]))
    files = natsorted(files)
    num_files = np.size(files,0)
    tiff_file_path = os.path.join(results_dir,folders[i],files[0])
    temp = io.imread(tiff_file_path).astype(float)
    num = len(files) #int(''.join(filter(str.isdigit, files[-1])))
    lenToUse = min(num,math.floor(len(time[i])/1000)) #Changed for ocamp testing 2024_08_05
    sz_x = tiffbatch*lenToUse
    sz_y = width
    tempfiber1 = np.zeros([sz_x, sz_y])
    tempfiber2 = np.zeros([sz_x, sz_y])
    peak = np.zeros(sz_x)
    for j in range(lenToUse):
    # for j in range(num_files-1):
        print(files[j])
        # Extract the number in the filename and convert to integer
        num = int(''.join(filter(str.isdigit, files[j])))
        tiff_file_path = os.path.join(results_dir,folders[i],files[j])
        temp = io.imread(tiff_file_path).astype(float)
        # background = np.mean(temp[:,0:20,0:200]) # Top corner left pixels are used for background subtraction
        for frame in range(temp.shape[0]):
            temp_rotated = cv.warpAffine(temp[frame,:,:],M1,(cols,rows))
            img = cv.warpAffine(temp_rotated,M2,(cols,rows))
            img = img #- background
            fiber1_m = np.mean(img[fiber1_location[1]:fiber1_location[0],:],axis=0)
            fiber2_m = np.mean(img[fiber2_location[1]:fiber2_location[0],:],axis=0)
            
            # # Use if running mouse
            temp_peak, _ = find_peaks(fiber2_m,height=200,distance=50)
            if len(temp_peak) > 1:
                max_peak = np.argmax(fiber2_m[temp_peak])
                temp_peak = temp_peak[max_peak]
            
            if frame==999 and (temp_peak is None):
                temp_peak = np.nan
            # print(temp_peak)
            # Use if running mouse with tdTomato
            # idx = np.where(fiber2_m > 300)
            # idx = idx[0][-1]
            # temp_peak = idx

            # # Use if running slide imaging
            # idx = np.where(fiber1_m > 10000)
            # idx = idx[0][-1]
            # temp_peak = idx

            fiber1_m = np.expand_dims(fiber1_m, axis=0)
            fiber2_m = np.expand_dims(fiber2_m, axis=0)
            if j == 0:
                tempfiber1[frame,:] = fiber1_m
                tempfiber2[frame,:] = fiber2_m
                peak[frame] = temp_peak
            else:
                tempfiber1[j*tiffbatch + frame,:] = fiber1_m
                tempfiber2[j*tiffbatch + frame,:] = fiber2_m
                peak[j*tiffbatch + frame] = temp_peak

    fiber1.append(tempfiber1)
    fiber2.append(tempfiber2)
    peaks.append(peak)

# Truncate time vectors to match the length of tiff files
for i in range(len(time)):
    if time[i].shape > fiber1[i].shape:
        time[i] = time[i][0:fiber1[i].shape[0]]
        
#%% Set order of lasers and interleave into individual laser channels

c = pd.read_hdf(results_dir/'pixel_to_nm.hdf5', key='Camera_pixel', more='r')
w = pd.read_hdf(results_dir/'pixel_to_nm.hdf5', key='Wavelength_nm', more='r')
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

#%% Interleave signals from each fiber into the five laser channels
L405_F1 = []
L445_F1 = []
L473_F1 = []
L514_F1 = []
L560_F1 = []
L405_F2 = []
L445_F2 = []
L473_F2 = []
L514_F2 = []
L560_F2 = []
time_405 = []
time_445 = []
time_473 = []
time_514 = []
time_560 = []

for i in range(len(folders)):
    L405_idx = np.where(laser_order[i]==405)
    L405_idx = np.array(L405_idx[0])
    F1_405 = fiber1[i][L405_idx,:]
    F2_405 = fiber2[i][L405_idx,:]
    temptime_405 = np.array(time[i][L405_idx])
    L445_idx = np.where(laser_order[i]==445)
    L445_idx = np.array(L445_idx[0])
    F1_445 = fiber1[i][L445_idx,:]
    F2_445 = fiber2[i][L445_idx,:]
    temptime_445 = np.array(time[i][L445_idx])
    L473_idx = np.where(laser_order[i]==488)
    L473_idx = np.array(L473_idx[0])
    F1_473 = fiber1[i][L473_idx,:]
    F2_473 = fiber2[i][L473_idx,:]
    temptime_473 = np.array(time[i][L473_idx])
    L514_idx = np.where(laser_order[i]==514)
    L514_idx = np.array(L514_idx[0])
    F1_514 = fiber1[i][L514_idx,:]
    F2_514 = fiber2[i][L514_idx,:]
    temptime_514 = np.array(time[i][L514_idx])
    L560_idx = np.where(laser_order[i]==561)
    L560_idx = np.array(L560_idx[0])
    F1_560 = fiber1[i][L560_idx,:]
    F2_560 = fiber2[i][L560_idx,:]
    temptime_560 = np.array(time[i][L560_idx])
    
    L405_F1.append(F1_405)
    L445_F1.append(F1_445)
    L473_F1.append(F1_473)
    L514_F1.append(F1_514)
    L560_F1.append(F1_560)
    L405_F2.append(F2_405)
    L445_F2.append(F2_445)
    L473_F2.append(F2_473)
    L514_F2.append(F2_514)
    L560_F2.append(F2_560)
    time_405.append(temptime_405)
    time_445.append(temptime_445)
    time_473.append(temptime_473)
    time_514.append(temptime_514)
    time_560.append(temptime_560)

#%% Concatenate signals from all frame folders into a single sequence
LCh405_F1 = np.concatenate(L405_F1)
LCh445_F1 = np.concatenate(L445_F1)
LCh473_F1 = np.concatenate(L473_F1)
LCh514_F1 = np.concatenate(L514_F1)
LCh560_F1 = np.concatenate(L560_F1)
LCh405_F2 = np.concatenate(L405_F2)
LCh445_F2 = np.concatenate(L445_F2)
LCh473_F2 = np.concatenate(L473_F2)
LCh514_F2 = np.concatenate(L514_F2)
LCh560_F2 = np.concatenate(L560_F2)
lasers = np.concatenate(laser_order)
timeseries_405 = np.concatenate(time_405)
timeseries_445 = np.concatenate(time_445)
timeseries_473 = np.concatenate(time_473)
timeseries_514 = np.concatenate(time_514)
timeseries_560 = np.concatenate(time_560)
full_time = np.concatenate(time)

#%% # Convert pixel data into wavelength data

LCh_405_F1 = np.zeros([LCh405_F1.shape[0], wavelength.shape[0]])
LCh_405_F2 = np.zeros([LCh405_F2.shape[0], wavelength.shape[0]])
LCh_445_F1 = np.zeros([LCh445_F1.shape[0], wavelength.shape[0]])
LCh_445_F2 = np.zeros([LCh445_F2.shape[0], wavelength.shape[0]])
LCh_473_F1 = np.zeros([LCh473_F1.shape[0], wavelength.shape[0]])
LCh_473_F2 = np.zeros([LCh473_F2.shape[0], wavelength.shape[0]])
LCh_514_F1 = np.zeros([LCh514_F1.shape[0], wavelength.shape[0]])
LCh_514_F2 = np.zeros([LCh514_F2.shape[0], wavelength.shape[0]])
LCh_560_F1 = np.zeros([LCh560_F1.shape[0], wavelength.shape[0]])
LCh_560_F2 = np.zeros([LCh560_F2.shape[0], wavelength.shape[0]])

for px in range(0,wavelength.shape[0]):
    LCh_405_F1[:,px] = LCh405_F1[:,camera_px[px]-Xoffset]
    LCh_405_F2[:,px] = LCh405_F2[:,camera_px[px]-Xoffset]
    LCh_445_F1[:,px] = LCh445_F1[:,camera_px[px]-Xoffset]
    LCh_445_F2[:,px] = LCh445_F2[:,camera_px[px]-Xoffset]
    LCh_473_F1[:,px] = LCh473_F1[:,camera_px[px]-Xoffset]
    LCh_473_F2[:,px] = LCh473_F2[:,camera_px[px]-Xoffset]
    LCh_514_F1[:,px] = LCh514_F1[:,camera_px[px]-Xoffset]
    LCh_514_F2[:,px] = LCh514_F2[:,camera_px[px]-Xoffset]
    LCh_560_F1[:,px] = LCh560_F1[:,camera_px[px]-Xoffset]
    LCh_560_F2[:,px] = LCh560_F2[:,camera_px[px]-Xoffset]


#%% Regress laser noise from 445 and 473 channels

# High pass filter the signal
# Scale noise linearly to signal 
# Subtract scaled noise from signal
# Add signal back to low frequency component of original signal

from scipy import signal

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Low pass filter the signal
fps = 150.25/5 # 24 for pre-2024
noise_px = 473
pass_freq = 5

def denoise_signal(sig, noise_px, pass_freq, fps):
    new_sig = np.zeros(sig.shape)
    for i in range(0,len(sig[0,:])):
        lowfilt_sig = butter_lowpass_filter(sig[:,i],pass_freq,fps)
        noise = sig[:,noise_px-400]
        A = np.vstack([noise, np.ones(len(noise))]).T
        [x,y] = np.linalg.lstsq(A, lowfilt_sig, rcond=None)[0]
        scl_noise = x*noise + y
        final_sig = lowfilt_sig - scl_noise + np.mean(lowfilt_sig)
        new_sig[:,i] = final_sig
    return new_sig

# noise_px = 445
# denoised_signal_445_F1 = denoise_signal(LCh_445_F1, noise_px, pass_freq, fps)
# denoised_signal_445_F2 = denoise_signal(LCh_445_F2, noise_px, pass_freq, fps)
# noise_px = 473
# denoised_signal_473_F1 = denoise_signal(LCh_473_F1, noise_px, pass_freq, fps)
# denoised_signal_473_F2 = denoise_signal(LCh_473_F2, noise_px, pass_freq, fps)

# Low pass filter the other signals

def lowpass_signal(sig, pass_freq, fps):
    new_sig = np.zeros(sig.shape)
    for i in range(0,len(sig[0,:])):
        lowfilt_sig = butter_lowpass_filter(sig[:,i],pass_freq,fps)
        new_sig[:,i] = lowfilt_sig
    return new_sig

pass_freq = 5
filtered_405_F1 = lowpass_signal(LCh_405_F1,pass_freq,fps)
filtered_405_F2 = lowpass_signal(LCh_405_F2,pass_freq,fps)
filtered_445_F1 = lowpass_signal(LCh_445_F1,pass_freq,fps)
filtered_445_F2 = lowpass_signal(LCh_445_F2,pass_freq,fps)
filtered_473_F1 = lowpass_signal(LCh_473_F1,pass_freq,fps)
filtered_473_F2 = lowpass_signal(LCh_473_F2,pass_freq,fps)
filtered_514_F1 = lowpass_signal(LCh_514_F1,pass_freq,fps)
filtered_514_F2 = lowpass_signal(LCh_514_F2,pass_freq,fps)
filtered_560_F1 = lowpass_signal(LCh_560_F1,pass_freq,fps)
filtered_560_F2 = lowpass_signal(LCh_560_F2,pass_freq,fps)

LCh_405_F1 = filtered_405_F1
LCh_405_F2 = filtered_405_F2
LCh_445_F1 = filtered_445_F1
LCh_445_F2 = filtered_445_F2
LCh_473_F1 = filtered_473_F1
LCh_473_F2 = filtered_473_F2
LCh_514_F1 = filtered_514_F1
LCh_514_F2 = filtered_514_F2
LCh_560_F1 = filtered_560_F1
LCh_560_F2 = filtered_560_F2

#%% Save preprocessed data in hdf5 file

data_preprocessed = {'Time_405':timeseries_405, 'Time_445':timeseries_445, 'Time_473':timeseries_473, 'Time_514':timeseries_514, 
                        'Time_560':timeseries_560, 'Full_TimeStamps':full_time,'Lasers':lasers, 'Wavelength':wavelength, 
                        'Channel_405_F1':LCh_405_F1, 'Channel_445_F1':LCh_445_F1, 'Channel_473_F1':LCh_473_F1,
                        'Channel_514_F1':LCh_514_F1, 'Channel_560_F1':LCh_560_F1, 'Channel_405_F2':LCh_405_F2,
                        'Channel_445_F2':LCh_445_F2, 'Channel_473_F2':LCh_473_F2, 'Channel_514_F2':LCh_514_F2,
                        'Channel_560_F2':LCh_560_F2}
for key in data_preprocessed.keys():
    print(f'\n{key}')
    print(data_preprocessed[key])

# Write a new hdf5 file with all keys in data_preprocessed
hf = h5py.File(results_dir / 'hsfp_data_preprocessed.hdf5','w')
for key in data_preprocessed.keys():
    hf.create_dataset(key, data = data_preprocessed[key])
hf.close()




