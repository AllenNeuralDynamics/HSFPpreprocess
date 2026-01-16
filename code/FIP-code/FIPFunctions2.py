# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:34:39 2026

@author: carrie.stine
"""
import os
import csv
import numpy as  np
import pylab as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
import glob
import re

#%%  original preprocess functions

def tc_crop(tc, nFrame2cut):
    tc_cropped = tc[nFrame2cut:]
    return tc_cropped
    
def tc_medfilt(tc, kernelSize):
    tc_filtered = medfilt(tc, kernel_size=kernelSize)
    return tc_filtered

def tc_lowcut(tc, sampling_rate):
    b,a = butter(2, 9, btype='low', fs=sampling_rate)
    tc_filtered = filtfilt(b,a, tc)
    return tc_filtered

def tc_polyfit(tc, sampling_rate, degree):
    time_seconds = np.arange(len(tc)) /sampling_rate 
    coefs = np.polyfit(time_seconds, tc, deg=degree)
    tc_poly = np.polyval(coefs, time_seconds)
    return tc_poly

def tc_slidingbase(tc, sampling_rate):
    b,a = butter(2, 0.0001, btype='low', fs=sampling_rate)
    tc_base = filtfilt(b,a, tc, padtype='even')
    return tc_base

def tc_dFF(tc, tc_base, b_percentile):
    tc_dFoF = tc/tc_base
    sort = np.sort(tc_dFoF)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    tc_dFoF = tc_dFoF - b_median
    return tc_dFoF

def tc_filling(tc, nFrame2cut):
    tc_filled = np.append(np.ones([nFrame2cut,1])*tc[0], tc)
    return tc_filled
    

#Preprocessing total function

def tc_preprocess(tc, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile):
    tc_cropped = tc_crop(tc, nFrame2cut)
    tc_filtered = medfilt(tc_cropped, kernel_size=kernelSize)
    tc_filtered = tc_lowcut(tc_filtered, sampling_rate)
    tc_poly = tc_polyfit(tc_filtered, sampling_rate, degree)
    tc_estim = tc_filtered - tc_poly
    tc_base = tc_slidingbase(tc_filtered, sampling_rate)
    #tc_dFoF = tc_dFF(tc_filtered, tc_base, b_percentile)
    tc_dFoF = tc_dFF(tc_estim, tc_base, b_percentile)    
    tc_dFoF = tc_filling(tc_dFoF, nFrame2cut)
    return tc_dFoF




#%% load_fip_data
def load_fip_data(anal_dir):
    """
    Loads FIP raw data (Iso, G, R) and associated timestamp files.
    Returns: data1, data2, data3, subjectID, TSdict
    """
    
    def read_csv_to_numpy(filepath, skip_header=True):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            if skip_header:
                next(reader)
            return np.array([row for row in reader]).astype(np.float32)

    # 1. Determine File Paths & Subject ID
    try: 
        file1 = glob.glob(os.path.join(anal_dir, "FIP_DataIso_*"))[0]
        base_path = anal_dir
    except IndexError:
        # Fallback to 'fib' directory if not found in anal_dir
        base_path = os.path.join(anal_dir[:-8], 'fib')
        file1 = glob.glob(os.path.join(base_path, "FIP_DataIso_*"))[0]
    
    file2 = glob.glob(os.path.join(base_path, "FIP_DataG_*"))[0]
    file3 = glob.glob(os.path.join(base_path, "FIP_DataR_*"))[0]
    
    # Extract Subject ID (assuming standard folder structure)
    # Using os.sep to be cross-platform compatible
    path_parts = anal_dir.split(os.sep)
    subject_id = path_parts[3] if len(path_parts) > 3 else "Unknown"

    # 2. Load Main Data Files
    data1 = read_csv_to_numpy(file1)
    data2 = read_csv_to_numpy(file2)
    data3 = read_csv_to_numpy(file3)

    # 3. Load Timestamp Files into Dictionary
    ts_files = glob.glob(os.path.join(anal_dir, "TS_*"))
    ts_dict = {}

    for fullpath_i in ts_files:
        file_name = os.path.basename(fullpath_i)
        # Extract key from filename (e.g., TS_Lick_... -> Lick)
        match = re.search(r'TS_(.*?)_', file_name)
        key = match.group(1) if match else file_name
        
        # Check for header
        with open(fullpath_i, 'r') as f:
            try:
                has_header = csv.Sniffer().has_header(f.read(1024))
            except:
                has_header = False
        
        ts_dict[key] = read_csv_to_numpy(fullpath_i, skip_header=has_header)

    return data1, data2, data3, subject_id, ts_dict


#%% sync_and_time
def sync_and_time(data1, data2, data3, sampling_rate):
    """
    Truncates data arrays to the minimum common length and generates a time vector.
    
    Returns:
        data1, data2, data3 (truncated arrays)
        PMts (timestamp column from data2)
        time_seconds (generated time array)
    """
    # 1. Determine the shortest length across all data streams
    # (In case one stream was cut off early)
    Length = np.amin([len(data1), len(data2), len(data3)])

    # 2. Slice all arrays to match that minimum length
    data1 = data1[0:Length]  # iso
    data2 = data2[0:Length]  # signal (Green)
    data3 = data3[0:Length]  # stim (Red)

    # 3. Extract the original timestamps from the signal channel
    PMts = data2[:, 0]

    # 4. Generate the uniform time vector in seconds
    # math: index / sampling_rate
    time_seconds = np.arange(len(data1)) / sampling_rate
    
    return data1, data2, data3, PMts, time_seconds

#%% preprocess_all_channels

def preprocess_all_channels(data1, data2, data3, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile):
    """
    Loops through all ROI columns and applies tc_preprocess to each.
    
    Returns:
        Ctrl_dF_F, G_dF_F, R_dF_F (Arrays of shape [Frames, ROIs])
    """
    # 1. Determine dimensions
    # Total frames is the number of rows
    # Number of ROIs is total columns minus 1 (since index 0 is time)
    n_frames = data1.shape[0]
    n_rois = data1.shape[1] - 1
    
    # 2. Initialize output arrays (Frames x ROIs)
    Ctrl_dF_F = np.zeros((n_frames, n_rois))
    G_dF_F = np.zeros((n_frames, n_rois))
    R_dF_F = np.zeros((n_frames, n_rois))

    # 3. Loop through each ROI column
    # data[:, ii+1] skips the timestamp column at index 0
    for ii in range(n_rois):
        Ctrl_dF_F[:, ii] = tc_preprocess(data1[:, ii+1], nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
        G_dF_F[:, ii]    = tc_preprocess(data2[:, ii+1], nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
        R_dF_F[:, ii]    = tc_preprocess(data3[:, ii+1], nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
        
    return Ctrl_dF_F, G_dF_F, R_dF_F


#%% extract_trial_frames
def extract_trial_frames(TSdict, reference_timestamps):
    """
    Converts event timestamps into frame indices based on the photometry clock.
    reference_timestamps: data1[:, 0] (the photometry time column)
    """
    TSFramesdict = {}

    for k, event_times in TSdict.items():
        # event_times is likely shape [N, 1] or [N, 2], we take column 0
        times = event_times[:, 0]
        
        # We use searchsorted for speed, then adjust to find the NEAREST value
        # This replaces the np.argmin(np.abs(...)) loop and is much faster
        indices = np.searchsorted(reference_timestamps, times)
        
        # Ensure indices are within bounds for the "nearest" check
        indices = np.clip(indices, 1, len(reference_timestamps) - 1)
        left_val = reference_timestamps[indices - 1]
        right_val = reference_timestamps[indices]
        
        # Choose the index that is actually closer in time
        closer_to_left = np.abs(times - left_val) < np.abs(times - right_val)
        final_indices = np.where(closer_to_left, indices - 1, indices)
        
        TSFramesdict[k] = final_indices.astype(np.float32)

    return TSFramesdict


#%% load_pupil_data
def load_pupil_data(AnalDir, phot_start_time):
    """
    Loads Pupil Tracking and Camera data, syncs lengths, and aligns time to photometry.
    phot_start_time: data1[0, 0]
    """
    pupil_files = glob.glob(os.path.join(AnalDir, "PupilTracking*"))
    eyecam_files = glob.glob(os.path.join(AnalDir, "FaceEyeCamera*.csv"))

    if not pupil_files or not eyecam_files:
        return None, None

    # Load Pupil Data
    with open(pupil_files[0]) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data_Pupil = np.array([row for row in reader]).astype(np.float32)

    # Load Camera Timestamps
    with open(eyecam_files[0]) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data_EyeCam_time = np.array([row for row in reader]).astype(np.float32)

    # Sync lengths
    length = np.amin([len(data_EyeCam_time), len(data_Pupil)])
    data_EyeCam_time = data_EyeCam_time[:length]
    data_Pupil = data_Pupil[:length]

    # Align to photometry start time
    # We find the camera frame closest to the first photometry frame
    idx_align = np.argmin(np.abs(data_EyeCam_time - phot_start_time))
    
    # Normalize time: set photometry start to 0 and convert ms to s
    aligned_time = (data_EyeCam_time - data_EyeCam_time[idx_align]) / 1000.0
    
    return aligned_time, data_Pupil

#%% plot_whole_trace
def plot_whole_trace(time_seconds, Ctrl_dF_F, G_dF_F, R_dF_F, Roi2Vis, event_dict, ds_factor=10):
    """
    Plots a separate figure with downsampled traces and event overlays.
    event_dict: dictionary containing 'Lick', 'Reward', 'CS1', 'CS2', 'CS3' frames.
    """
    # 1. Downsample for performance
    t_ds = time_seconds[::ds_factor]
    
    # 2. Create the Figure
    plt.figure(figsize=(18, 10))
    
    # 3. Plot ROI Traces
    for i, roi_idx in enumerate(Roi2Vis):
        offset = i * 100
        # Plot signals (Control, Green, Red)
        plt.plot(t_ds, Ctrl_dF_F[::ds_factor, roi_idx]*100 - offset, color='blue', lw=1, alpha=0.7)
        plt.plot(t_ds, G_dF_F[::ds_factor, roi_idx]*100 - offset, color='green', lw=1, alpha=0.7)
        plt.plot(t_ds, R_dF_F[::ds_factor, roi_idx]*100 - offset, color='magenta', lw=1, alpha=0.7)
        # Baseline
        plt.plot(t_ds, np.zeros_like(t_ds) - offset, '--k', alpha=0.2)

    # 4. Plot Event Overlays (Vertical Bars)
    # Mapping event names to (frames, color, duration)
    meta = {
        'Reward': (event_dict.get('Reward', []), [0, 0.5, 1, 0.3], 2.0),
        'CS1':    (event_dict.get('CS1', []),    [1, 0, 0, 0.3], 1.0),
        'CS2':    (event_dict.get('CS2', []),    [0, 1, 0, 0.3], 1.0),
        'CS3':    (event_dict.get('CS3', []),    [1, 0, 1, 0.3], 1.0)
    }

    for label, (frames, color, dur) in meta.items():
        for f in frames:
            plt.axvspan(f/20, f/20 + dur, color=color, lw=0, label=label)

    # 5. Plot Licks
    licks = event_dict.get('Lick', [])
    if len(licks) > 0:
        plt.vlines(licks/20, ymin=50, ymax=100, color='black', alpha=0.4, label='Lick')

    # 6. Formatting
    plt.title(f"Whole Trace Analysis - Downsampled {ds_factor}x")
    plt.xlabel('Time (seconds)')
    plt.ylabel('dF/F (%) with ROI Offset')
    plt.xlim(t_ds[0], t_ds[-1])
    
    # Handle Legend (Avoid duplicates)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.12, 1))
    
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.show()
