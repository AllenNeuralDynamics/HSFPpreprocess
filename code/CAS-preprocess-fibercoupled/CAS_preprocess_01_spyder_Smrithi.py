# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:54:50 2025

@author: svc_aind_behavior
"""
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
import warnings

warnings.filterwarnings("ignore")

#%% variables
SAT_VAL = 7000 # saturation value of camera
FIBER_WIDTH = 40 # width of fiber (in pixels)

# store the session id
session_id = "815738_2025-11-25T11_19_02.6493184-08_00"

#%% Load calibration filenames and metadata

# point to the data directory
data_dir = r"C:\output_data\\"

# create a path to the session data
# path = '/root/capsule/data/HSFP_775510_2025-02-20_11-08-27'
path = os.path.join(data_dir,session_id)
print('session data path: ')
print(path)

#create a path to the calibration files
calib_path = os.path.join(path, r'fib\CalibrationFiles')
print('\ncalibration file path: ')
print(calib_path)

# print the names of all files in the 'CalibrationFiles' folder
files = os.listdir(calib_path)
print('\ncalibration files: ')
print(files)

# identify the calibration metadata files
metadata_file = glob.glob(os.path.join(calib_path,'*.csv'))
print('\ncsv file names: ')
print(metadata_file)

# filter for the csv starting with session_params
session_params_file = [f for f in metadata_file if 'session_params' in os.path.basename(f)]
print('\nmetadata file name: ')
print(session_params_file[0])

# convert the session_params metadata csv into a Pandas DataFrame
metadata = pd.read_csv(session_params_file[0])

# create a path to the calibration tiffs
tiff_dir = os.path.join(calib_path,'Tiffs')
tiff_files = os.listdir(tiff_dir)
tiff_files = np.array(natsorted(tiff_files))
print('\nsorted calibration tiff files: ')
print(tiff_files)

#%% Load calibration image
# take the first tiff file and build its full path
tiff_file_path = os.path.join(tiff_dir,tiff_files[0])
print('calibration tiff file path: ')
print(tiff_file_path)

# open the tiff using python imaging library
tiff = Image.open(tiff_file_path)
image_sequence = []

# Extract all frames from the first multi-frame tiff into an image_sequence list of numpy arrays (each represents one frame/image)
while True:
    try:
        image_sequence.append(np.array(tiff))
        tiff.seek(tiff.tell() + 1)
    except EOFError:
        break
print('\nnumber of images in tiff: ')
len(image_sequence)

#%% Average all frames together
img = np.array(image_sequence) # convert the list of frames into a 3D numpy array (num_frames, height, width)
img2d = img.mean(axis=0) # average across all frames (axis=0) to create a single 2D image
print('averaged image dimensions (height, width):')
print(img2d.shape) # display the dimensions of the averaged image (height, width)

# extract camera ROI metadata (x and y length + positioning)
width = metadata.Width[0] # camera pixels in x direction
print('\n# of pixels in x direction (width):')
print(width)

Xoffset = metadata.XOffset[0] # start of camera pixels in x direction
print('\nstarting position of pixels in x direction:')
print(Xoffset)

height = metadata.Height[0] # camera pixels in y direction
print('\n# of pixels in y direction (height):')
print(height)

Yoffset = metadata.YOffset[0] # start of camera pixels in y direction
print('\nstarting position of pixels in y direction:')
print(Yoffset)

#%% plot averaged image
img_to_unskew = np.array(img2d)
f,ax = plt.subplots(figsize=(6,6))
i = ax.imshow(img_to_unskew, aspect='auto', vmin=0, vmax=SAT_VAL)
ax.set(xlabel='Camera pixels', ylabel='Camera pixels', title='Calibration Image')
ax.grid(False)
f.colorbar(i,ax=ax)
plt.show()

#%% Extract rotation of image to flatten spectrum
# Find laser positions (horizontally)
h_orig_line = np.mean(img_to_unskew,axis=0) # average the image vertically, laser spots will appear as peaks in a horizontal 1D profile
h_orig_laser_pix, _ = find_peaks(h_orig_line, height=2000, distance=50) #find peaks brighter than 2000 and at least 50 pixels apart

# Find laser positions (vertically)
count = 0 # Initialize counter for laser number
v_orig_laser_pix = np.zeros(np.size(h_orig_laser_pix)) # Initialize array to store vertical positions of each laser

for l in h_orig_laser_pix: # Run the loop for each laser's horizontal position 'l'
    v_orig_line = img_to_unskew[:,l].copy() # Extract a vertical column from the original calibration image at each position 'l'
    v_orig_line[v_orig_line<SAT_VAL] = 0 # Set all pixels below the sat_val threshold to 0 to isolate the brightest spots
    idx = np.nonzero(np.diff(v_orig_line))[0] # Find indices where bright spots transition to 0 (top and bottom edges of the laser spot)
    v_orig_laser_pix[count] = np.round((idx[0]+idx[-1])/2) # Find the laser's vertical midpoint by averaging its top and bottom edges
    count +=1 # Increase the laser counter by 1
    
print(h_orig_laser_pix)
print(v_orig_laser_pix)


#%% Plot midpoints of each laser
f, ax = plt.subplots(figsize=(6,2))
ax.plot(h_orig_laser_pix,v_orig_laser_pix,'o')
ax.set(xlabel='Camera pixels', ylabel='Camera pixels', title='Midpoint of laser/fiber in unskewed image')
plt.show()

#%% Calculate rotation angle needed to unskew calibration image
# Find the tilt angle between the first and second-to-last laser
theta_r = np.rad2deg(np.arctan((v_orig_laser_pix[-2]-v_orig_laser_pix[0])/(h_orig_laser_pix[-2]-h_orig_laser_pix[0])))
# v_orig_laser_pix = vertical distance (rise); h_orig_laser_pix = horizontal distance (run); arctan(rise/run) = angle in radians
print('Rotation angle:')
print(theta_r)

# Rotate image by the calculated angle using openCV
rows,cols = img_to_unskew.shape 
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),theta_r,1) # create a rotation transformation matrix
# (cols-1)/2.0,(rows-1)/2.0) = image center; theta_r = tilt angle; 1 = scale (no scaling, just rotation)
img_rotated = cv.warpAffine(img_to_unskew,M,(cols,rows)) # apply the rotation transformation to unskew the image

# Confirm that rotation did not alter overall height/width of the image
print('\noriginal image dimensions (height, width):')
print(img2d.shape) 
print('\nunskewed image dimensions (height, width):')
print(img_rotated.shape) 

# Plot the rotated/unskewed image
f,ax = plt.subplots(figsize=(6,6))
i = ax.imshow(img_rotated, aspect='auto', vmin=0, vmax=SAT_VAL)
ax.set(xlabel='Camera pixels', ylabel='Camera pixels', title='Rotated Image')
ax.grid(False)
f.colorbar(i,ax=ax)
plt.show()

#%% # Check position of lasers in rotated image
# Find laser positions (horizontally)
h_line = np.mean(img_rotated,axis=0)
h_laser_pix, _ = find_peaks(h_line, height=2000, distance=50)

# Find laser positions (vertically)
count = 0
v_laser_pix = np.zeros(np.size(h_laser_pix))
for l in h_laser_pix:
    v_line = img_rotated[:,l].copy()
    v_line[v_line<SAT_VAL] = 0
    idx = np.nonzero(np.diff(v_line))[0]
    v_laser_pix[count] = np.round((idx[0]+idx[-1])/2)
    count +=1
v_laser_pix = v_laser_pix.astype(int)

# Plot midpoints of each laser after rotation
f, ax = plt.subplots(figsize=(6,2))
ax.plot(h_laser_pix,v_laser_pix,'o')
ax.set(xlabel='Camera pixels', ylabel='Camera pixels', title='Midpoint of laser/fiber after rotation')
plt.show()

print('\nHorizontal midpoint of each laser:')
print(h_laser_pix)
print('\nVertical midpoint of each laser:')
print(v_laser_pix)

#%% definitions for affine transformation

def analyze_laser(img_rotated, h_laser_pix, use_laser, sat_val, fiber_width=60):
    """Finds and plots the top/bottom edges of a fiber for a given laser."""
    
    # --- Vertical boundaries ---
    v_line = img_rotated[:, h_laser_pix[use_laser]].copy() # extract one vertical column of pixel intensities at the laser's x-position
    v_line[v_line < sat_val] = 0 # set values below the intensity threshold to zero
    idx = np.nonzero(np.diff(v_line))[0] # find locations where intensity changes (edge of laser)
    v_width = [int(idx[0]), int(idx[-1])] # store the top and bottom y-coordinates of the laser
    
    # --- Horizontal boundaries (top and bottom) ---
    h_width = []
    for position, label, offset in zip(
        [v_width[0] + 5, v_width[1] - 5],
        ["Top", "Bottom"],
        ["h_top", "h_bot"]
    ):
        # Extract horizontal cross-section
        h_line = img_rotated[
            position,
            h_laser_pix[use_laser] - fiber_width : h_laser_pix[use_laser] + fiber_width
        ].copy()
        h_line[h_line < sat_val] = 0
        h_edges = np.nonzero(np.diff(h_line))[0]
        h_edge = int(h_edges[0])

        # --- Plot for visualization --- 
        f, ax = plt.subplots(figsize=(4, 2))
        ax.plot(h_line)
        ax.set(
            xlabel="Camera pixels",
            ylabel="Saturation intensity",
            title=f"Laser {use_laser + 1} Fiber {label}"
        )
        plt.show()

        # Convert to absolute coordinates
        h_width.append(int(h_laser_pix[use_laser] - fiber_width + h_edge))

    # --- Print summary ---
    print(f"Laser {use_laser + 1}")
    print(f"  Vertical pixel indices (top, bottom): {v_width}")
    print(f"  Horizontal pixel indices (top left, bottom left): {h_width}\n\n")
    
    return v_width, h_width






def check_laser_post_affine(img, h_laser_pix, use_laser, sat_val, fiber_width=60):
    """ Check the top and bottom of a laser fiber in the post-affine transformed image.
    Parameters
    ----------
    img : 2D array
        The affine-transformed image (img_final)
    h_laser_pix : list or array
        X positions of lasers
    use_laser : int
        Index of the laser to analyze
    fiber_width : int
        Half-width of the horizontal crop around the laser
    sat_val : int
        Threshold value to zero out background pixels
    
    Returns
    -------
    v_width : list of int
        Vertical pixel indices [top, bottom] of the laser
    h_width : list of int
        Horizontal pixel indices [top-left, bottom-left] of the laser
    """
    
    # --- Vertical coordinates ---
    v_line = img[:, h_laser_pix[use_laser]].copy()          # vertical slice at laser x-position
    v_line[v_line < sat_val] = 0                     # threshold low intensity
    idx = np.nonzero(np.diff(v_line))[0]             # edges along y-axis
    v_width = [int(idx[0]), int(idx[-1])]            # top and bottom y-coordinates

    # --- Horizontal coordinates ---
    # Top edge
    h_line_top = img[v_width[0]+5, h_laser_pix[use_laser]-fiber_width:h_laser_pix[use_laser]+fiber_width].copy()
    h_line_top[h_line_top < sat_val] = 0
    # h_top_idx = np.nonzero(np.diff(h_line_top))[0][0]  # first edge
    # h_top = int(h_laser_pix[use_laser] - fiber_width + h_top_idx)
    h_top_idx = np.nonzero(np.diff(h_line_top))[0]
    h_top = int(h_top_idx[0])

    # Bottom edge
    h_line_bot = img[v_width[1]-5, h_laser_pix[use_laser]-fiber_width:h_laser_pix[use_laser]+fiber_width].copy()
    h_line_bot[h_line_bot < sat_val] = 0
    # h_bot_idx = np.nonzero(np.diff(h_line_bot))[0][0]
    # h_bot = int(h_laser_pix[use_laser] - fiber_width + h_bot_idx)
    h_bot_idx = np.nonzero(np.diff(h_line_bot))[0]
    h_bot = int(h_bot_idx[0])

    #h_width = [h_top, h_bot]
    # Convert to absolute coordinates
    h_width = [int(h_laser_pix[use_laser]-fiber_width + h_top),
               int(h_laser_pix[use_laser]-fiber_width + h_bot)]

    # --- Plotting ---
    for line, name in zip([h_line_top, h_line_bot], ["Top", "Bottom"]):
        f, ax = plt.subplots(figsize=(4,2))
        ax.plot(line)
        ax.set(xlabel='Camera pixels', ylabel='Saturation intensity', title=f'Laser {use_laser + 1} Fiber {name}')
        plt.show()
        
     # --- Print summary ---
    print(f"\nLaser {use_laser + 1}")
    print(f"  Vertical pixel indices (top, bottom): {v_width}")
    print(f"  Horizontal pixel indices (top left, bottom left): {h_width}")

    return v_width, h_width

#%% Find fiber top and bottom edges
# Find the coordinates of the top and bottom of the fiber image for the given lasers
# for laser_idx in [1, 2]:
#     analyze_laser(img_rotated, h_laser_pix, laser_idx, sat_val, fiber_width=60)

# Find fiber coordinates for laser 1
use_laser_1 = 0
v_width, h_width = analyze_laser(img_rotated, h_laser_pix, use_laser_1, SAT_VAL, fiber_width=FIBER_WIDTH)

# Find fiber coordinates for laser 3
use_laser_2 = 2
v_width2, h_width2 = analyze_laser(img_rotated, h_laser_pix, use_laser_2, SAT_VAL, fiber_width=FIBER_WIDTH)

print(v_width, h_width, v_width2, h_width2)

#%%  Perform the affine transformation
# v_width[0] = bottom of fiber, v_width[1] = top of fiber, h_width[0] = left side of fiber, h_width[1] = right side of fiber

# Specify the location of three points in the source image
pt1 = [h_width[0], v_width[0]] # top-left corner of 1st fiber
pt2 = [h_width[1], v_width[1]] # bottom-left corner of 1st fiber
pt3 = [h_width2[0], v_width2[0]] # top-left corner of 2nd fiber

# Specify where you want those 3 points to map onto in the transformed image
pt4 = [h_width[1], v_width[0]] # map top-left of 1st fiber to bottom-left x-position (shear/stretch top in x direction)
pt5 = [h_width[1], v_width[1]] # map bottom-left of 1st fiber to bottom-left x-position (no transformation)
pt6 = [h_width2[1], v_width2[0]] # map top-left of 2nd fiber to bottom-left x-position (shear/stretch top in x direction)


print(pt1, pt2, pt3, pt4, pt5, pt6)

rows,cols = img_rotated.shape # get height and width of the image to define the output size in warpAffine
pts1 = np.float32([pt1, pt2, pt3]) # source points (from original image)
pts2 = np.float32([pt4, pt5, pt6]) # destination points (where those points should end up in the output image)
M = cv.getAffineTransform(pts1,pts2) # compute a 2x3 affine transformation matrix
img_final = cv.warpAffine(img_rotated,M,(cols,rows)) # apply the transfomration to the whole image

f,ax = plt.subplots(figsize=(6,6))
i = ax.imshow(img_final, aspect='auto', vmin=0, vmax=SAT_VAL)
ax.set(xlabel='Camera pixels', ylabel='Camera pixels', title='Rotated and affine transformed')
ax.grid(False)
f.colorbar(i,ax=ax)
plt.show()

f,ax = plt.subplots(figsize=(8,2))
i = ax.imshow(img_final, aspect='auto', vmin=0, vmax=SAT_VAL)
ax.set(xlabel='Camera pixels', ylabel='Camera pixels',  title='Rotated and affine transformed', xlim=[0,2048], ylim=[pt5[1]+50,pt6[1]-50])
ax.grid(False)
f.colorbar(i,ax=ax)
plt.show()

#%% # Check that transformation aligned laser top and bottom
# Check the first laser used
check_laser_post_affine(img_final, h_laser_pix, use_laser_1, SAT_VAL, fiber_width=FIBER_WIDTH)

# Check the second laser used
check_laser_post_affine(img_final, h_laser_pix, use_laser_2, SAT_VAL, fiber_width=FIBER_WIDTH)


#%% # Find pixels that correspond to fiber 1 and fiber 2
# Create a 1D array representing mean brightness at each x-position
h_line_new = np.mean(img_final,axis=0) # calculate the average intensity along each column of the final image
h_laser_pix_new, _ = find_peaks(h_line_new, height=2000, distance=50) # identify x-positions of peaks above 2000 and at least 100 pixels apart
v_line = img_final[:,h_laser_pix_new[use_laser_2]].copy()
v_line[v_line < SAT_VAL] = 0
idx = np.nonzero(np.diff(v_line))[0]
v_width = [int(idx[0]), int(idx[-1])]
print(f"Laser {use_laser_2+1} \nVertical pixel indices (top, bottom): {v_width}")

# Find laser positions (vertically)
count = 0
v_laser_pix_new = np.zeros(np.size(h_laser_pix_new))
for l in h_laser_pix_new:
    v_line = img_final[:,l].copy()
    v_line[v_line<SAT_VAL] = 0
    idx = np.nonzero(np.diff(v_line))[0]
    v_laser_pix_new[count] = np.round((idx[0]+idx[-1])/2)
    count +=1
v_laser_pix_new = v_laser_pix_new.astype(int)

# Adjust the top and bottom of each fiber slightly to define subregions inside the fiber
# Define vertical bounds of the first fiber
fiber1_top = v_width[0] + 20 # 20 pixels below the top edge
fiber1_bottom = v_width[0] + 50 - 10 # 40 pixels below the top edge
# Define vertical bounds of the second fiber
fiber2_top = v_width[1] - 50 + 10 # 40 pixels above the bottom edge
fiber2_bottom = v_width[1] - 20 # 20 pixels above the bottom edge

# Store the vertical boundaries of each fiber
fiber1 = [fiber1_top, fiber1_bottom]
fiber2 = [fiber2_top, fiber2_bottom]
print(f"\nLaser {use_laser_2+1}")
print(f"Fiber 1 boundaries (top, bottom): {fiber1}")
print(f"Fiber 2 boundaries (top, bottom): {fiber2}")

print(h_laser_pix_new)
print(v_laser_pix_new)

#%% # Plot the sections of the calibration image containing each fiber
# Fiber 1 location
f,ax = plt.subplots(figsize=(8,2))
i = ax.imshow(img_final, aspect='auto', vmin=0, vmax=SAT_VAL)
ax.set(xlabel='Camera pixels', ylabel='Camera pixels',  title='Fiber 1 (rotated and affine transformed)', xlim=[0,2048], ylim=[fiber1[1],fiber1[0]])
ax.grid(False)
f.colorbar(i,ax=ax)
plt.show()

# Fiber 2 location
f,ax = plt.subplots(figsize=(8,2))
i = ax.imshow(img_final, aspect='auto', vmin=0, vmax=SAT_VAL)
ax.set(xlabel='Camera pixels', ylabel='Camera pixels',  title='Fiber 2 (rotated and affine transformed)', xlim=[0,2048], ylim=[fiber2[1],fiber2[0]])
ax.grid(False)
f.colorbar(i,ax=ax)
plt.show()

#%% # Create 1x3 subplots to show raw, rotated, and transformed image

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Left: Original image with midpoints ---
im0 = axes[0].imshow(img_to_unskew, aspect='auto', vmin=0, vmax=SAT_VAL)
axes[0].plot(h_orig_laser_pix, v_orig_laser_pix, 'ro', markersize=5, label='Laser midpoints')
axes[0].set(title='Calibration Image - Raw', xlabel='Camera pixels', ylabel='Camera pixels')
axes[0].legend()
#cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
#cbar0.set_label('Pixel Intensity')

# --- Middle: Rotated image with midpoints ---
im1 = axes[1].imshow(img_rotated, aspect='auto', vmin=0, vmax=SAT_VAL)
axes[1].plot(h_laser_pix, v_laser_pix, 'ro', markersize=5, label='Laser midpoints')
axes[1].set(title='Calibration Image - Rotated', xlabel='Camera pixels', 
            #ylabel='Camera pixels'
           )
axes[1].yaxis.set_visible(False)
axes[1].legend()
#cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04) # uncomment to give plot its own colorbar
#cbar1.set_label('Pixel Intensity')

# --- Right: Final image with midpoints ---
im2 = axes[2].imshow(img_final, aspect='auto', vmin=0, vmax=SAT_VAL)
axes[2].plot(h_laser_pix_new, v_laser_pix_new, 'ro', markersize=5, label='Laser midpoints')

# Add horizontal lines for fiber boundaries
axes[2].axhline(fiber1[0], color='red', linestyle='--', linewidth=2, label='Fiber 1 Boundaries')
axes[2].axhline(fiber1[1], color='red', linestyle='--', linewidth=2, 
                #label='Fiber1 Bottom'
               )
axes[2].axhline(fiber2[0], color='blue', linestyle='--', linewidth=2, label='Fiber 2 Boundaries')
axes[2].axhline(fiber2[1], color='blue', linestyle='--', linewidth=2, 
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


#%% Save the affine transformed points and fiber locations
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

#%% Save transformed image for wavelength calibration
img_final = img_final.astype(int)

#create a path to the calibration files
results_path = os.path.join(path, 'fib')
results_dir = Path(results_path)
results_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists

output_file = results_dir / "CalibrationImage.tiff"
cv.imwrite(str(output_file), img_final)

with open(results_dir / 'calibration_new.txt','w') as f:
    f.write(f'rot_tform_thetaR = {theta_r}\n')
    f.write(f'aff_tform_pt1 = {pt1}\n')
    f.write(f'aff_tform_pt2 = {pt2}\n')
    f.write(f'aff_tform_pt3 = {pt3}\n')
    f.write(f'aff_tform_pt4 = {pt4}\n')
    f.write(f'aff_tform_pt5 = {pt5}\n')
    f.write(f'aff_tform_pt6 = {pt6}\n')
    f.write(f'fiber1_pixels = {fiber1}\n')
    f.write(f'fiber2_pixels = {fiber2}\n')
    f.write(f'calib_Xoffset = {Xoffset}\n')


