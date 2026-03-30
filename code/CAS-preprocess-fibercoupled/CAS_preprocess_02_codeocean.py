# CAS_preprocess_02_pixel_to_nm.py
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from PIL import Image
from pathlib import Path
# from config import load_config
import argparse
from datetime import datetime as dt
import h5py
import ast

# Lasers used for calibration
lasers = np.array([0.594, 0.514, 0.488, 0.445, 0.405])

# Physical constants
SELLMEIER_COEFFS = (1.73759695, 0.313747356, 1.89878101)
SELLMEIER_TERMS = (0.013188707, 0.0623068142, 155.23629)
THETA_I = 60.8  # angle of incidence (degrees)
ALPHA = 60      # prism apex angle (degrees)
SAT_VAL = 7000  # value of laser saturation


# Refractive index + angle of deviation helper functions
def sellmeier_index(wavelength):
    """Compute refractive index (n) from wavelength in microns."""
    B1, B2, B3 = SELLMEIER_COEFFS
    C1, C2, C3 = SELLMEIER_TERMS
    wav2 = wavelength**2
    return np.sqrt(1 + (B1*wav2)/(wav2 - C1) + (B2*wav2)/(wav2 - C2) + (B3*wav2)/(wav2 - C3))

def deviation_angle(wavelength, theta_i=THETA_I, alpha=ALPHA):
    """Compute deviation angle for a given wavelength."""
    n = sellmeier_index(wavelength)
    term = (np.sin(np.deg2rad(alpha)) * np.sqrt(n**2 - np.sin(np.deg2rad(theta_i))**2)
            - np.sin(np.deg2rad(theta_i)) * np.cos(np.deg2rad(alpha)))
    return theta_i - alpha + np.rad2deg(np.arcsin(term))

def linear_wave(x, a, b):
    """Linear fit function between deviation angle and pixel value."""
    return a * x + b


# Calibration image helper function
def find_laser_pixels(image_path, height_thresh=2000, dist_thresh=50, sat_val=SAT_VAL):
    """Find x-pixel positions of laser peaks in calibration image."""
    I = Image.open(image_path)
    im = np.array(I)
    im_line = np.mean(im, axis=0)
    peaks, _ = find_peaks(im_line, height=height_thresh, distance=dist_thresh)
    
    f,ax = plt.subplots(figsize=(6,6))
    ax.imshow(im, vmin=0, vmax=sat_val)
    ax.set(title='Calibration Image', xlabel='Camera pixels')
    ax.grid(False)
    
    plt.figure(figsize=(6,2))
    plt.plot(im_line)
    plt.plot(peaks, im_line[peaks], 'x')
    plt.title('Laser Peaks in Calibration Image')
    plt.show()
    
    return peaks


# Calibration fitting function
def fit_wavelength_to_pixels(lasers, pixel_positions):
    """Fit wavelength-pixel relationship using linear model."""
    theta_D = deviation_angle(lasers)
    popt, _ = curve_fit(linear_wave, theta_D, pixel_positions)
    return popt, theta_D


# LUT building function
def generate_lut(popt, calib_file, wavelength_range=(0.4, 0.7, 0.001)):
    wavelength = np.arange(*wavelength_range)
    theta_D = deviation_angle(wavelength)
    pixel_value = linear_wave(theta_D, *popt)

    # Safe parsing of calibration.txt
    calib_dict = {}
    with open(calib_file, 'r') as f:
        for line in f:
            name, value = line.strip().split(' = ')
            calib_dict[name] = ast.literal_eval(value)  # Safe conversion

    pixel_value_new = calib_dict['calib_Xoffset'] + np.round(pixel_value).astype(int)
    lut_df = pd.DataFrame({'Camera_pixel': pixel_value_new, 'Wavelength_nm': (1000*wavelength).astype(int)})
    return lut_df, pixel_value, theta_D



# Plotting function
def plot_calibration_results(lasers, laser_pix, wavelength, pixel_value):
    """Plot calibration fit and deviation angle mapping."""
    plt.figure(figsize=(8,3))
    plt.plot(pixel_value, 1000*wavelength, 'b-', label='Fitted Curve')
    plt.plot(laser_pix, 1000*lasers, 'ro', label='Measured Lasers')
    plt.xlabel('Camera pixel')
    plt.ylabel('Wavelength (nm)')
    plt.legend()
    plt.show()
    
    
    
#%% Main    
# Main execution block
if __name__ == '__main__': # ensures this code only runs if the script is executed directly, and not when imported as a module
    # parser = argparse.ArgumentParser(
    #     description="Calibrate prism dispersion and generate wavelength-to-pixel LUT"
    # )
    # parser.add_argument(
    #     "--results_dir",
    #     type=str,
    #     default="/results/",
    #     help="Directory containing the calibration image and calibration.txt file",
    # )
    # args = parser.parse_args()
    # results_dir = Path(args.results_dir)
    
    # Define session_id
    # session_id = "815738_2025-11-25T11_19_02.6493184-08_00" # NEED TO CORRECT FOR CODE OCEAN
    
    print("Starting HSFP image calibration processing step 1...")
        
    # Settings
    data_dir = "/data"
    results_base_dir = "/results"
    
    # Get list of session IDs (subfolders in data_dir)
    session_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not session_ids:
        raise FileNotFoundError(f"No session subfolders found in {data_dir}")
    
    for session_id in session_ids:
        print(f"Processing session: {session_id}")

                 
        calib_image = results_base_dir / 'CalibrationImage.tiff'
        calib_file = results_base_dir / 'calibration.txt'

        laser_pix = find_laser_pixels(calib_image)

    popt, theta_D = fit_wavelength_to_pixels(lasers, laser_pix)
    print(f"Fit parameters: a={popt[0]:.4f}, b={popt[1]:.4f}")

    wavelength = np.arange(0.4, 0.7, 0.001)
    pixel_value = linear_wave(deviation_angle(wavelength), *popt)
    plot_calibration_results(lasers, laser_pix, wavelength, pixel_value)

    lut, pixel_value_wave, theta_D_wave = generate_lut(popt, calib_file)
    print(lut.head())
    
    hdf5_file = results_dir / 'pixel_to_nm.hdf5'
    store = pd.HDFStore(hdf5_file, mode='a')
    for col in lut.columns:
        lut[col].to_hdf(store, key=col, mode='a')
    store.close()
    
    with h5py.File(hdf5_file, 'r') as f:
        print("HDF5 keys:", list(f.keys()))
    
    print("Calibration processing step 2 complete.")