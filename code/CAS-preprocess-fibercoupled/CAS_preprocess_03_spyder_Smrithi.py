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
#%%
# store the session id
session_id = 'BigTiffs2025-02-20T11_08_27.6009600-08_00'

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
    timestamp = metadata.CameraTimestampSeconds + 10**-6*metadata.CameraTimestampMicroSeconds
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
print(width)
Xoffset = metadata.XOffset[0]
Xoffset = Xoffset.astype(int)
print(Xoffset)
height = metadata.Height[0]
print(height)
Yoffset = metadata.YOffset[0]
Yoffset = Yoffset.astype(int)
print(Yoffset)

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

# Read through each Tiff directory, exrtact tiff files, and unskew images
fiber1 = []
fiber2 = []
peaks = []
tiffbatch = 1000
for i in range(len(folders)):
    print(folders[i])
    files = os.listdir(os.path.join(results_dir,folders[i]))
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
            temp_peak, _ = find_peaks(fiber2_m,height=200,distance=200)
            if len(temp_peak) > 1:
                max_peak = np.argmax(fiber2_m[temp_peak])
                temp_peak = temp_peak[max_peak]
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
                tempfiber1[num*tiffbatch + frame,:] = fiber1_m
                tempfiber2[num*tiffbatch + frame,:] = fiber2_m
                peak[num*tiffbatch + frame] = temp_peak

    fiber1.append(tempfiber1)
    fiber2.append(tempfiber2)
    peaks.append(peak)

# Truncate time vectors to match the length of tiff files
for i in range(len(time)):
    if time[i].shape > fiber1[i].shape:
        time[i] = time[i][0:fiber1[i].shape[0]]
