# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:34:39 2026

@author: carrie.stine
"""
import os
import csv
import numpy as  np
#import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
import glob
import re
import pandas as pd
import h5py
import seaborn as sns
from datetime import datetime
import scipy.stats as stats
from scipy.signal import find_peaks
import random

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
    tc_base = tc_slidingbase(tc_filtered, sampling_rate) # should this be tc_estim?
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
    # DEPRECATED - this is Kenta's version which has folders organized in a 
    # different structure than what I use
    # path_parts = anal_dir.split(os.sep)
    # subject_id = path_parts[3] if len(path_parts) > 3 else "Unknown"
    
    parent_folder = os.path.basename(os.path.dirname(anal_dir))
    # Split by underscore: ['FIP', '000000', 'yyyy-MM-dd', 'HH-mm-ss']
    # Index [1] is always the 6-digit ID
    try:
        subject_id = parent_folder.split('_')[1]
    except IndexError:
        subject_id = "Unknown"
        
        
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
def get_event_frames(TSdict, reference_timestamps):
    """
    Dynamically processes all keys found in TSdict.
    Converts timestamps to frame indices for any event type present.
    """
    event_frames = {}
    
    for key, data in TSdict.items():
        # 1. Skip if data is None or completely empty
        if data is None or data.size == 0:
            event_frames[key] = np.array([], dtype=np.float32)
            continue
            
        # 2. Extract timestamps (Handle 1D or 2D arrays)
        # Using .ndim check to prevent IndexError
        times = data[:, 0] if data.ndim > 1 else data
        
        # 3. Fast nearest-neighbor search
        # Find where 'times' would fit into the 'reference_timestamps'
        indices = np.searchsorted(reference_timestamps, times)
        
        # Keep indices within the valid range of the reference array
        indices = np.clip(indices, 1, len(reference_timestamps) - 1)
        
        # Compare left and right neighbors to find the absolute nearest frame
        left_val = reference_timestamps[indices - 1]
        right_val = reference_timestamps[indices]
        
        closer_to_left = np.abs(times - left_val) < np.abs(times - right_val)
        final_indices = np.where(closer_to_left, indices - 1, indices)
        
        event_frames[key] = final_indices.astype(np.float32)
            
    return event_frames


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
def plot_whole_trace(time_seconds, Ctrl_dF_F, G_dF_F, R_dF_F, Roi2Vis, event_dict, 
                     subjectID, AnalDir, StimPeriod, ds_factor=10, pupil_data=None):
    
    # 1. Downsample for plotting performance
    t_ds = time_seconds[::ds_factor]
    
    # 2. Setup Figure (Matching original 16x16 or 18x10)
    fig = plt.figure(figsize=(18, 10))
    
    # 3. Plot ROI Traces (Matching original blue/green/magenta style)
    ii_ROI = 0 # Initialize for pupil tracking offset logic
    for i, roi_idx in enumerate(Roi2Vis):
        ii_ROI = i
        offset = ii_ROI * 100
        
        plt.plot(t_ds, Ctrl_dF_F[::ds_factor, roi_idx]*100 - offset, 'blue', label='Control' if i==0 else "")
        plt.plot(t_ds, G_dF_F[::ds_factor, roi_idx]*100 - offset, 'green', label='Green' if i==0 else "")
        plt.plot(t_ds, R_dF_F[::ds_factor, roi_idx]*100 - offset, 'magenta', label='Red' if i==0 else "")
        plt.plot(t_ds, np.zeros(len(t_ds)) - offset, '--k', alpha=0.5)

    # 4. Plot Licks (Matching marker=3 style)
    licks = event_dict.get('Lick', [])
    if len(licks) > 0:
        plt.plot(licks/20, np.ones(len(licks))*100, marker=3, markersize=10, 
                 color=[0, 0, 0, 0.5], ls='None', label='Lick')

    # 5. Define Specific Styles for known events
    # (Frames, Color, Duration)
    meta = {
        'Reward': (event_dict.get('Reward', []), [0, 0, 1, 0.4], StimPeriod),
        'CS1':    (event_dict.get('CS1', []),    [1, 0, 0, 0.4], 1.0),
        'CS2':    (event_dict.get('CS2', []),    [0, 1, 0, 0.4], 1.0),
        'CS3':    (event_dict.get('CS3', []),    [1, 0, 1, 0.4], 1.0)
    }

    # Plot specific event spans
    for label, (frames, color, dur) in meta.items():
        if len(frames) > 0:
            # OPTIMIZATION: Only label the VERY FIRST span for the legend
            # This prevents Matplotlib from tracking thousands of legend handles
            plt.axvspan(frames[0]/20, frames[0]/20 + dur, color=color, lw=0, label=label)
            
            # Plot the rest without labels (much faster)
            for f in frames[1:]:
                plt.axvspan(f/20, f/20 + dur, color=color, lw=0)

    # 6. DYNAMICALLY plot any OTHER keys found in event_dict
    # This catches "AirPuff", "Tone", etc., that weren't in the list above
    known_keys = list(meta.keys()) + ['Lick']
    other_colors = plt.cm.get_cmap('tab10') # Use a color map for variety
    
    for i, (key, frames) in enumerate(event_dict.items()):
        if key not in known_keys and len(frames) > 0:
            c = other_colors(i % 10)
            for f in frames:
                plt.axvspan(f/20, f/20 + 0.5, color=c, alpha=0.2, lw=0)
            plt.axvspan(0, 0, color=c, alpha=0.2, label=key)

    # 7. Pupil Tracking (Optional)
    if pupil_data is not None:
        p_time, p_values = pupil_data # Unpack tuple
        plt.plot(p_time, p_values - (ii_ROI)*100 - 50, color=[0.4, 0.4, 0.4], label='PupilDiam.')

    # 8. Formatting (Matching original Title/Labels)
    plt.xlabel('Time (seconds)')
    plt.ylabel('dF/F (%)')
    date_str = os.path.basename(os.path.dirname(AnalDir))
    plt.title(f"SubjectID: {subjectID}  Date: {date_str}")
    plt.xlim([0, time_seconds[-1]])
    plt.grid(True)
    
    # Legend deduplication
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.1, 1))
    
    plt.tight_layout()
    
    # return the figure as an object for saving
    return fig
    
#%% #%% define PSTH functions (for multiple traces)
def PSTHmaker(TC, Stims, preW, postW):
    
    cnt = 0
    
    for ii in range(len(Stims)):
        if Stims[ii] - preW >= 0 and  Stims[ii] + postW < len(TC):
            
            A = int(Stims[ii]-preW) 
            B = int(Stims[ii]+postW)
            
            if cnt == 0:
                PSTHout = TC[A:B,:]
                cnt = 1
            else:
                PSTHout = np.dstack([PSTHout,TC[A:B,:]])
        else:
            if cnt == 0:
                PSTHout = np.zeros(preW+postW)
                cnt = 1
            #else:
                #PSTHout = np.dstack([PSTHout, np.zeros(preW+postW)])
    return PSTHout


#%% Define PSTH baseline subtraction (multi)
#dim0:trial, dim1:time
def PSTH_baseline(PSTH, preW):

    for ii in range(np.shape(PSTH)[2]):
        
        Trace_this = PSTH[:, :, ii]
        Trace_this_base = Trace_this[0:preW,:]
        Trace_this_subtracted = Trace_this - np.mean(Trace_this_base,axis=0)        
        
        if ii == 0:
            PSTHbase = Trace_this_subtracted
        else:
            PSTHbase = np.dstack([PSTHbase,Trace_this_subtracted])
    
    return PSTHbase

    
#%% get_trial_indices
def get_trial_indices(AnalDir):
    """
    Identifies rewarded (R) and unrewarded (UR) trial indices for CS1, CS2, and CS3.
    Handles standard sessions, reversal sessions, and missing CSV fallbacks.
    """
    # Initialize empty arrays to ensure variables exist even if logic is skipped
    results = {
        'Mat_CS1': np.array([]), 'Mat_CS2': np.array([]), 'Mat_CS3': np.array([]),
        'RewardedCS1ind': np.array([]), 'RewardedCS2ind': np.array([]), 'RewardedCS3ind': np.array([]),
        'UnRewardedCS1ind': np.array([]), 'UnRewardedCS2ind': np.array([]), 'UnRewardedCS3ind': np.array([])
    }

    trial_file = glob.glob(os.path.join(AnalDir, "TrialN_*"))
    reversal_file = glob.glob(os.path.join(AnalDir, "Trial_Reversal_*"))

    # --- SCENARIO 1 & 2: CSV Data Available ---
    if trial_file:
        df = pd.read_csv(trial_file[0])
        
        if reversal_file:
            # SCENARIO 2: Reversal Logic
            df_rev = pd.read_csv(reversal_file[0])
            Sw = df_rev['TrialSwitched'][0]
            
            # Pre-Switch / Standard
            results['Mat_CS1'] = np.where((df['TrialType']<=10) & (df['TrialType']>=1) & (df['TrialNumber']<=Sw))[0]
            results['Mat_CS2'] = np.where((df['TrialType']<=20) & (df['TrialType']>=11))[0]
            results['Mat_CS3'] = np.where((df['TrialType']<=30) & (df['TrialType']>=21) & (df['TrialNumber']<=Sw))[0]
            
            CS1R = np.where((df['TrialType']==1) & (df['TrialNumber']<=Sw))[0]
            CS1UR = np.where((df['TrialType']<=10) & (df['TrialType']>=2) & (df['TrialNumber']<=Sw))[0]
            CS2R = np.where((df['TrialType']<=15) & (df['TrialType']>=11))[0]
            CS2UR = np.where((df['TrialType']<=20) & (df['TrialType']>=16))[0]
            CS3R = np.where((df['TrialType']<=29) & (df['TrialType']>=21) & (df['TrialNumber']<=Sw))[0]
            CS3UR = np.where((df['TrialType']==30) & (df['TrialNumber']<=Sw))[0]

            # Post-Switch (Identity Swapping)
            results['Mat_CS1'] = np.append(results['Mat_CS1'], np.where((df['TrialType']<=30) & (df['TrialType']>=21) & (df['TrialNumber']>Sw))[0])
            results['Mat_CS3'] = np.append(results['Mat_CS3'], np.where((df['TrialType']<=10) & (df['TrialType']>=1) & (df['TrialNumber']>Sw))[0])
            
            CS1R = np.append(CS1R, np.where((df['TrialType']<=29) & (df['TrialType']>=21) & (df['TrialNumber']>Sw))[0])
            CS1UR = np.append(CS1UR, np.where((df['TrialType']==30) & (df['TrialNumber']>Sw))[0])
            CS3R = np.append(CS3R, np.where((df['TrialType']==1) & (df['TrialNumber']>Sw))[0])
            CS3UR = np.append(CS3UR, np.where((df['TrialType']<=10) & (df['TrialType']>=2) & (df['TrialNumber']>Sw))[0])
            
        else:
            # SCENARIO 1: Standard Trial Logic
            results['Mat_CS1'] = np.where((df['TrialType']<=10) & (df['TrialType']>=1))[0]
            results['Mat_CS2'] = np.where((df['TrialType']<=20) & (df['TrialType']>=11))[0]
            results['Mat_CS3'] = np.where((df['TrialType']<=30) & (df['TrialType']>=21))[0]
            
            CS1R = np.where(df['TrialType']==1)[0]
            CS1UR = np.where((df['TrialType']<=10) & (df['TrialType']>=2))[0]
            CS2R = np.where((df['TrialType']<=15) & (df['TrialType']>=11))[0]
            CS2UR = np.where((df['TrialType']<=20) & (df['TrialType']>=16))[0]
            CS3R = np.where((df['TrialType']<=29) & (df['TrialType']>=21))[0]
            CS3UR = np.where(df['TrialType']==30)[0]

        # Map Reward status to Mat_CS indices
        results['RewardedCS1ind'] = np.where(np.isin(results['Mat_CS1'], CS1R))[0]
        results['RewardedCS2ind'] = np.where(np.isin(results['Mat_CS2'], CS2R))[0]
        results['RewardedCS3ind'] = np.where(np.isin(results['Mat_CS3'], CS3R))[0]
        results['UnRewardedCS1ind'] = np.where(np.isin(results['Mat_CS1'], CS1UR))[0]
        results['UnRewardedCS2ind'] = np.where(np.isin(results['Mat_CS2'], CS2UR))[0]
        results['UnRewardedCS3ind'] = np.where(np.isin(results['Mat_CS3'], CS3UR))[0]

    return results    
    
    
#%% generate_all_psths
def generate_all_psths(G_dF_F, R_dF_F, Ctrl_dF_F, event_frames, trial_idx, preW=100, postW=300):
    """
    Generates baseline-subtracted PSTHs for all CS trial types across all channels.
    trial_idx: the dictionary returned by get_trial_indices()
    """
    psth_results = {}
    
    # Define mapping of Trial Types to their indices and corresponding Frame arrays
    # Format: { 'CategoryName': (Frames_Array, Index_Array) }
    mapping = {
        'CS1R':  (event_frames['CS1'], trial_idx['RewardedCS1ind']),
        'CS1UR': (event_frames['CS1'], trial_idx['UnRewardedCS1ind']),
        'CS2R':  (event_frames['CS2'], trial_idx['RewardedCS2ind']),
        'CS2UR': (event_frames['CS2'], trial_idx['UnRewardedCS2ind']),
        'CS3R':  (event_frames['CS3'], trial_idx['RewardedCS3ind']),
        'CS3UR': (event_frames['CS3'], trial_idx['UnRewardedCS3ind']),
    }

    channels = {
        'G': G_dF_F * 100,
        'R': R_dF_F * 100,
        'C': Ctrl_dF_F * 100
    }

    for name, (frames, indices) in mapping.items():
        # Only process if we have indices for this trial type
        if len(indices) > 0:
            # Subset the frames based on trial classification (R vs UR)
            target_stims = frames[indices.astype(int)]
            
            for ch_key, ch_data in channels.items():
                # 1. Create Raw PSTH
                raw = PSTHmaker(ch_data, target_stims, preW, postW)
                
                # 2. Subtract Baseline and store in dictionary
                # Key example: 'G_CS3R_base'
                psth_results[f'{ch_key}_{name}_base'] = PSTH_baseline(raw, preW)
        else:
            # Fill with None or empty if no trials exist for this session
            for ch_key in channels.keys():
                psth_results[f'{ch_key}_{name}_base'] = None

    return psth_results

#%% pool_rois_in_psths
def pool_rois_in_psths(psth_data, Roi2Vis):
    pooled_data = {}
    
    for key, data in psth_data.items():
        if data is not None:
            # 1. Slice out only the ROIs you want to combine
            # data is [Time, ROI, Trial]
            subset = data[:, Roi2Vis, :] 
            
            # 2. Get dimensions
            n_time, n_roi, n_trials = subset.shape
            
            # 3. Reshape: merge ROI (dim 1) and Trial (dim 2)
            # New shape: [n_time, 1, (n_roi * n_trials)]
            pooled_data[key] = subset.reshape(n_time, 1, -1)
        else:
            pooled_data[key] = None
            
    return pooled_data

#%% sort_timestamps_by_cs
def sort_timestamps_by_cs(TSdict, trial_data):
    """
    Creates a dictionary of timestamps for each CS for rewarded trials only.
    """
    TSdict_CSrewarded = {}
    cs_types = ["CS1", "CS2", "CS3"]
    
    # Extract only the primary clock (Col 0) for matching logic
    # Using .size check to handle empty arrays safely
    all_rewards = TSdict["Reward"][:, 0] if TSdict["Reward"].size > 0 else np.array([])
    all_licks = TSdict["Lick"][:, 0] if TSdict["Lick"].size > 0 else np.array([])

    for cs in cs_types:
        cs_onsets_list = []
        reward_ts_list = []
        first_lick_ts_list = []

        # 1. Access the raw 2D array and rewarded indices
        cs_raw = TSdict.get(cs, np.array([]))
        r_indices = trial_data.get(f"Rewarded{cs}ind", [])

        # 2. Only proceed if the CS array is not empty and has rewarded trials
        if cs_raw.size > 0 and len(r_indices) > 0:
            # Slicing the first column of the 2D array at the rewarded indices
            rewarded_onsets = cs_raw[r_indices, 0]

            for onset in rewarded_onsets:
                # Find the first reward timestamp strictly after CS onset
                future_rewards = all_rewards[all_rewards > onset]
                
                if future_rewards.size > 0:
                    rew_t = future_rewards[0]
                    
                    # Find the first lick timestamp strictly after reward delivery
                    future_licks = all_licks[all_licks > rew_t]
                    
                    if future_licks.size > 0:
                        lick_t = future_licks[0]
                        
                        # Sync all three timestamps for this specific trial
                        cs_onsets_list.append(onset)
                        reward_ts_list.append(rew_t)
                        first_lick_ts_list.append(lick_t)

        # Convert to 1D numpy arrays 
        cs_arr = np.array(cs_onsets_list)
        rew_arr = np.array(reward_ts_list)
        lick_arr = np.array(first_lick_ts_list)
        
        # 3. Create dictionary entries
        # If no trials met the criteria, these result in empty arrays
        TSdict_CSrewarded[cs] = cs_arr
        TSdict_CSrewarded[f"{cs}Reward"] = rew_arr
        TSdict_CSrewarded[f"{cs}FirstLick"] = lick_arr
        
        # Calculate Reaction Time (Lick - Reward)
        TSdict_CSrewarded[f"{cs}ReactionTime"] = lick_arr - rew_arr

    return TSdict_CSrewarded


#%% calculate_fip_peaks
def calculate_fip_peaks(psth_data, TSdict_CSrewarded, sampling_rate, preW, PeakWindow):
    """
    Calculates peaks in provided PeakWindow (relative to CS onset) and determines 
    peak latency relative to CS, Reward, and Lick. Uses TSdict_CSrewarded to determine 
    search windows per trial.
    """
    results = {}
    
    win_start_idx = int(preW + (PeakWindow[0] * sampling_rate))
    win_end_idx = int(preW + (PeakWindow[1] * sampling_rate))
    
    for key, data in psth_data.items():
        if data is None or not key.startswith(('G_', 'R_')):
            continue
            
        # Parse Key (e.g., 'G_CS3R_base' -> cs_type = 'CS3')
        # We only calculate reward/lick peaks for rewarded ('R_base') keys
        parts = key.split('_')
        trial_type = parts[1] # 'CS3R', 'CS1UR', etc.
        cs_type = trial_type[:3] # 'CS1', 'CS2', 'CS3'
        
        # 1. FIND THE SIGNAL PEAK in the specified window (All trials)
        # Standard window: 0s to 5s after CS onset
        subset = data[win_start_idx:win_end_idx, :, :]
        
        if subset.size == 0:
            continue
        
        # Get index of max relative to the window start
        peak_idx_in_window = np.argmax(subset, axis=0)
        
        # Convert to time relative to CS Onset (Time 0)
        # (Peak Index + Window Start Index - CS Onset Index) / Sampling Rate
        peak_times_rel_to_cs = (peak_idx_in_window + win_start_idx - preW) / sampling_rate
        
        results[f"{key}_peak_mag"] = np.max(subset, axis=0)
        results[f"{key}_peak_cs_lat"] = peak_times_rel_to_cs
        
        

        # 2. CALCULATE RELATIVE LATENCIES (rewarded trials only)
        if 'R_base' in key and 'UR_base' not in key and cs_type in TSdict_CSrewarded:
            beh = TSdict_CSrewarded
            
            # These are also 1D arrays of time relative to CS onset
            reward_offsets = (beh[f"{cs_type}Reward"] - beh[cs_type]) / 1000.0
            lick_offsets = (beh[f"{cs_type}FirstLick"] - beh[cs_type]) / 1000.0

            # We need to broadcast the 1D behavior offsets to match (ROIs, Trials)
            # peak_times_rel_to_cs is (ROIs, Trials)
            # reward_offsets is (Trials,)
            
            # Latency = (Time of Peak) - (Time of Event)
            # If Peak is at 2s and Reward is at 3s, Latency = -1s (Peak happened before Reward)
            results[f"{key}_peak_rew_lat"] = peak_times_rel_to_cs - reward_offsets[np.newaxis, :]
            results[f"{key}_peak_lick_lat"] = peak_times_rel_to_cs - lick_offsets[np.newaxis, :]
            
    return results


#%% plot_latency_comparison
def plot_latency_comparison(results, trial_type='CS3R', roi_idx=0):
    """
    Plots histograms of latencies relative to CS, Reward, and Lick
    to compare signal alignment.
    """
    # Extract the data from the results dictionary
    cs_lat = results.get(f'G_{trial_type}_base_peak_cs_lat', [])
    rew_lat = results.get(f'G_{trial_type}_base_peak_rew_lat', [])
    lick_lat = results.get(f'G_{trial_type}_base_peak_lick_lat', [])

    # Check if we have data (selecting specific ROI)
    if len(cs_lat) == 0:
        print(f"No data found for {trial_type}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f'Latency Alignment Comparison: {trial_type} (ROI {roi_idx})', fontsize=16)

    # Plot CS Latency
    sns.histplot(cs_lat[roi_idx, :], binwidth=0.05, ax=axes[0], color='skyblue', kde=True)
    axes[0].set_title('Relative to CS Onset')
    axes[0].set_xlabel('Time (s)')

    # Plot Reward Latency
    sns.histplot(rew_lat[roi_idx, :], binwidth=0.05, ax=axes[1], color='salmon', kde=True)
    axes[1].set_title('Relative to Reward')
    axes[1].set_xlabel('Time (s)')

    # Plot Lick Latency
    sns.histplot(lick_lat[roi_idx, :], binwidth=0.05, ax=axes[2], color='green', kde=True)
    axes[2].set_title('Relative to First Lick')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()
#%% plot_roi_psth_summary
def plot_roi_psth_summary(psths, subjectID, Roi2Vis, sampling_rate, StimPeriod, preW=100, trial_type='CS3R'):
    """
    Creates a grid where:
    - Each ROW is a different ROI.
    - The FINAL ROW is the Pooled/Combined ROIs.
    - Left Column: Green + Control
    - Right Column: Red + Control
    - Y-axes are matched within each ROI (row).
    """
    num_rois = len(Roi2Vis)
    total_rows = num_rois + 1  # Add an extra row for combined data
    
    fig, axes = plt.subplots(total_rows, 2, figsize=(10, 4 * total_rows), sharex=True)
    
    # Handle sizing indexing (subplots returns 1D array if only 1 row total)
    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # --- 1. Plot Individual ROIs ---
    for i, roi_idx in enumerate(Roi2Vis):
        ax_green = axes[i, 0]
        ax_red = axes[i, 1]
        
        _plot_psth_row(psths, roi_idx, ax_green, ax_red, sampling_rate, StimPeriod, preW, trial_type, f'ROI {roi_idx}')

    # --- 2. Plot Combined/Pooled Row ---
    # Call your pooling function internally for the ROIs in Roi2Vis
    pooled_psths = pool_rois_in_psths(psths, Roi2Vis)
    
    ax_green_comb = axes[-1, 0]
    ax_red_comb = axes[-1, 1]
    
    # Since pool_rois_in_psths collapses data into index 0
    _plot_psth_row(pooled_psths, 0, ax_green_comb, ax_red_comb, sampling_rate, StimPeriod, preW, trial_type, 'Combined ROIs')

    # --- 3. Final Formatting ---
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')

    fig.suptitle(f'{subjectID} Summary PSTH: {trial_type}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


#%% _plot_psth_row
def _plot_psth_row(psths, idx, ax_g, ax_r, sampling_rate, StimPeriod, preW, trial_type, title_prefix):
    """Internal helper to plot the Green and Red columns for a single grid row."""
    g_data = psths.get(f'G_{trial_type}_base')
    r_data = psths.get(f'R_{trial_type}_base')
    c_data = psths.get(f'C_{trial_type}_base')

    # Green + Control
    if g_data is not None and c_data is not None:
        PSTHplot_single_roi(g_data[:, idx, :], 'green', [0, 0.8, 0], 'Green', preW, sampling_rate, ax_g, StimPeriod, trial_type=trial_type)
        PSTHplot_single_roi(c_data[:, idx, :], 'blue', [0, 0, 0.8], 'Ctrl', preW, sampling_rate, ax_g, StimPeriod, trial_type=trial_type)
    ax_g.set_title(f'{title_prefix}: Green + Ctrl')
    ax_g.set_ylabel('dF/F (%)')

    # Red + Control
    if r_data is not None and c_data is not None:
        PSTHplot_single_roi(r_data[:, idx, :], 'magenta', [0.8, 0, 0.8], 'Red', preW, sampling_rate, ax_r, StimPeriod, trial_type=trial_type)
        PSTHplot_single_roi(c_data[:, idx, :], 'blue', [0, 0, 0.8], 'Ctrl', preW, sampling_rate, ax_r, StimPeriod, trial_type=trial_type)
    ax_r.set_title(f'{title_prefix}: Red + Ctrl')
    ax_r.set_ylabel('dF/F (%)')
    
    # Match Y-axes for the row
    y1_min, y1_max = ax_g.get_ylim()
    y2_min, y2_max = ax_r.get_ylim()
    common_min, common_max = min(y1_min, y2_min), max(y1_max, y2_max)
    ax_g.set_ylim(common_min, common_max)
    ax_r.set_ylim(common_min, common_max)
    
#%% PSTHplot_single_roi
def PSTHplot_single_roi(PSTH_subset, MainColor, SubColor, LabelStr, preW, sampling_rate, ax, StimPeriod, trial_type='CS3R'):
    time_x = np.arange(PSTH_subset.shape[0]) / sampling_rate - preW / sampling_rate
    mean_trace = np.mean(PSTH_subset, axis=1)
    n_trials = PSTH_subset.shape[1]
    sem = np.std(PSTH_subset, axis=1) / np.sqrt(PSTH_subset.shape[1])
    
    # --- 1. Dynamic CS Color Logic ---
    # Match the colors from your meta dict: CS1=Red, CS2=Green, CS3=Magenta
    cs_colors = {
        'CS1': [1, 0, 0, 0.2], # Red
        'CS2': [0, 1, 0, 0.2], # Green
        'CS3': [1, 0, 1, 0.2]  # Magenta
    }
    # Get the base name (first 3 chars) to pick the color
    base_type = trial_type[:3] 
    chosen_color = cs_colors.get(base_type, [0.5, 0.5, 0.5, 0.2]) # Default grey if not found

    # Plot CS Span
    ax.axvspan(0, 1.0, color=chosen_color, lw=0, label=base_type if LabelStr in ['Green', 'Red'] else "")

    # --- 2. Conditional Reward Logic ---
    # Only plot if 'R' is in the name and it's NOT 'UR'
    if 'R' in trial_type and 'UR' not in trial_type:
        ax.axvspan(2.0, 2.0 + StimPeriod, color=[0, 0, 1, 0.2], lw=0, label='Reward' if LabelStr in ['Green', 'Red'] else "")

    # --- 3. Plot Data ---
    full_label = f"{LabelStr} (n={n_trials})"
    ax.plot(time_x, mean_trace, label=full_label, color=MainColor, lw=2)
    ax.fill_between(time_x, mean_trace - sem, mean_trace + sem, facecolor=SubColor, alpha=0.3)
    
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.2)
    ax.legend(loc='upper right', fontsize='x-small')
    
    
#%% plot_time_to_peak_summary
def plot_time_to_peak_summary(psths, subjectID, Roi2Vis, sampling_rate, preW, trial_type='CS3R', search_window=[0, 5.0]):
    """
    Calculates and plots the Time to Peak for Green and Red channels.
    - Each ROW corresponds to the ROIs in Roi2Vis + one Combined row.
    - Bars represent the mean time-to-peak; points represent individual trials.
    """
    num_rois = len(Roi2Vis)
    total_rows = num_rois + 1
    
    fig, axes = plt.subplots(total_rows, 1, figsize=(6, 4 * total_rows), sharex=True)
    if total_rows == 1: axes = [axes]

    # Define the search window in indices
    start_idx = int(preW + (search_window[0] * sampling_rate)) 
    end_idx = int(preW + (search_window[1] * sampling_rate))

    # --- 1. Process Individual ROIs ---
    for i, roi_idx in enumerate(Roi2Vis):
        _plot_peak_bars(psths, roi_idx, axes[i], trial_type, start_idx, end_idx, sampling_rate, f'ROI {roi_idx}')

    # --- 2. Process Combined Row ---
    pooled_psths = pool_rois_in_psths(psths, Roi2Vis)
    _plot_peak_bars(pooled_psths, 0, axes[-1], trial_type, start_idx, end_idx, sampling_rate, 'Combined ROIs')

    # Formatting
    axes[-1].set_xlabel('Time to Peak (s)')
    fig.suptitle(f'{subjectID} Time to Peak Summary: {trial_type}\n(Window: {search_window[0]}-{search_window[1]}s)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def _plot_peak_bars(psths, idx, ax, trial_type, start_idx, end_idx, sampling_rate, title_prefix):
    """Helper to calculate peaks and draw the bar/dot plot for one row."""
    g_data = psths.get(f'G_{trial_type}_base')
    r_data = psths.get(f'R_{trial_type}_base')

    channel_data = {'Green': g_data, 'Red': r_data}
    colors = {'Green': 'green', 'Red': 'magenta'}
    y_positions = [1, 0] # Green on top, Red on bottom
    
    peaks_found = False

    for label, data in channel_data.items():
        if data is not None:
            # Slice: [TimeWindow, ROI_idx, AllTrials]
            # We search for the max value after Time 0
            trials_slice = data[start_idx:end_idx, idx, :]
            
            # Find index of max value for each trial
            # result is an array of indices relative to start_idx
            peak_indices = np.argmax(trials_slice, axis=0)
            
            # Convert indices to seconds
            peak_times = peak_indices / sampling_rate
            
            # Plot individual trial points (jittered for visibility)
            y_jitter = np.random.normal(y_positions[label=='Red'], 0.05, size=len(peak_times))
            ax.scatter(peak_times, y_jitter, color=colors[label], alpha=0.4, s=20)
            
            # Plot Mean Bar
            mean_peak = np.mean(peak_times)
            ax.barh(y_positions[label=='Red'], mean_peak, color=colors[label], alpha=0.2, height=0.6)
            ax.vlines(mean_peak, y_positions[label=='Red']-0.3, y_positions[label=='Red']+0.3, 
                      color=colors[label], lw=3, label=f"{label} (avg: {mean_peak:.2f}s)")
            
            peaks_found = True

    ax.set_yticks([1, 0])
    ax.set_yticklabels(['Green', 'Red'])
    ax.set_title(title_prefix)
    ax.set_xlim(0, 5) # Matches the search window
    if peaks_found:
        ax.legend(loc='upper right', fontsize='x-small')
        
        
        
        
#%% calculate_and_plot_rt
def calculate_and_plot_rt(ts_dict, subjectID, save_dir, save_figs=1):
    """
    Calculates latency from Reward timestamp to the first subsequent Lick.
    Generates a 3-panel summary PDF and returns the RT array.
    """
    # 1. Extract timestamps (assuming they are in seconds, adjust /1000 if not)
    RewardTime = ts_dict.get("Reward")
    LickTime = ts_dict.get("Lick")

    if RewardTime is None or LickTime is None:
        print("Skipping RT calculation: Reward or Lick timestamps missing.")
        return None


    RewardedLickFrames = np.empty(len(RewardTime))
    RewardRT = np.empty(len(RewardTime))

    # 2. Calculate Latency to first lick AFTER reward
    for ii in range(len(RewardTime)):
        # 1. Find the index of the lick closest in time to the reward
        idx = np.argmin(np.abs(LickTime[:, 0] - RewardTime[ii, 0]))
        
        # 2. If the closest lick was BEFORE or AT the reward, 
        # move to the next lick in the array.
        if LickTime[idx, 0] - RewardTime[ii, 0] <= 0:
            idx = idx + 1
            
        try:
            RewardedLickFrames[ii] = idx
            # Difference remains in milliseconds (assuming original TS units)
            RewardRT[ii] = LickTime[idx, 0] - RewardTime[ii, 0]
        except:
            # Handle cases where idx + 1 goes out of bounds
            print(f"skipped Reward index {ii}: lick index {idx} out of range")
            RewardRT[ii] = np.nan
            
            
            

    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: RT across trials
    axes[0].plot(RewardRT/1000)
    axes[0].set_ylabel('Reaction Time (s)')
    axes[0].set_xlabel('Reward #')
    axes[0].set_title('med RT:' + str(np.round(np.nanmedian(RewardRT / 1000), 4)) + ' (s)')

    # Panel 2: Distribution of all RTs
    axes[1].hist(RewardRT[~np.isnan(RewardRT)]/1000, bins=15, color='gray', edgecolor='black')
    axes[1].set_ylabel('Trial Count')
    axes[1].set_xlabel('Reaction Time (s)')
    axes[1].set_title('Full Distribution')

    # Panel 3: Zoomed distribution (Fast responses < 0.5s)
    fast_rts = RewardRT[RewardRT < 500]
    axes[2].hist(fast_rts/1000, bins=10, color='skyblue', edgecolor='black')
    axes[2].set_ylabel('Trial Count')
    axes[2].set_xlabel('Reaction Time < 0.5s (s)')
    axes[2].set_title(f'Fast Licks (n={len(fast_rts)})')

    plt.tight_layout()

    # 4. Save and Return
    if save_figs == 1:
        save_path = os.path.join(save_dir, f'{subjectID}_ReactionTime.pdf')
        fig.savefig(save_path, bbox_inches='tight')
        print(f"RT plot saved to: {save_path}")
    
    return RewardRT

#%% plot_comprehensive_trial_summary
# def generate_all_trial_summaries(psth_data, peak_results, TSdict_CSrewarded, 
#                                  roi_indices, sampling_rate, preW, save_dir, subjectID):
#     """Loops through all trial types and saves a comprehensive summary figure for each."""
#     trial_types = ['CS1R', 'CS1UR', 'CS2R', 'CS2UR', 'CS3R', 'CS3UR']
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     for tt in trial_types:
#         green_key = f"G_{tt}_base"
#         red_key = f"R_{tt}_base"
        
#         # Only trigger if data exists and is not None
#         if (green_key in psth_data and psth_data[green_key] is not None) or \
#            (red_key in psth_data and psth_data[red_key] is not None):
            
#             try:
#                 fig = plot_comprehensive_trial_summary(psth_data, peak_results, TSdict_CSrewarded, 
#                                                        tt, roi_indices, sampling_rate, preW, subjectID)
                
#                 save_filename = f"{subjectID}_Summary_{tt}_{timestamp}.png"
#                 fig.savefig(os.path.join(save_dir, save_filename), bbox_inches='tight', dpi=150)
#                 print(f"Generated: {save_filename}")
#             except Exception as e:
#                 print(f"Error on {tt}: {e}")

# def plot_comprehensive_trial_summary(psth_data, peak_results, TSdict_CSrewarded, 
#                                      trial_type, roi_indices, sampling_rate, preW, subjectID):
#     cs_name = trial_type[:3]
#     is_rewarded = 'UR' not in trial_type
#     cs_colors = {'CS1': [1, 0, 0, 0.4], 'CS2': [0, 1, 0, 0.4], 'CS3': [1, 0, 1, 0.4]}
#     colors = {'cs': cs_colors.get(cs_name), 'rew': [0, 0, 1, 0.4], 'lick': [0, 0, 0, 1.0]}

#     n_rois = len(roi_indices)
#     total_grid_rows = (n_rois + 1) * 3
#     fig = plt.figure(figsize=(30, 4 * (n_rois + 1)))
#     gs = gridspec.GridSpec(total_grid_rows, 9, figure=fig)
#     fig.suptitle(f"Subject: {subjectID} | Trial Type: {trial_type}", fontsize=22, fontweight='bold', y=1.02)

#     # Add 'combined' to the list of things to plot
#     plot_list = roi_indices + ['combined']

#     for r_idx, roi in enumerate(plot_list):
#         row_offset = r_idx * 3
        
#         for sig_idx, sig_prefix in enumerate(['G', 'R']):
#             col_start = sig_idx * 4
#             psth_key = f"{sig_prefix}_{trial_type}_base"
#             ctrl_key = f"{sig_prefix}_{trial_type}_ctrl"
            
#             if psth_key not in psth_data or psth_data[psth_key] is None:
#                 continue

#             # 1. PSTH
#             ax_psth = fig.add_subplot(gs[row_offset:row_offset+3, col_start])
#             _plot_psth_on_ax(ax_psth, psth_data, psth_key, psth_data.get(ctrl_key), 
#                              sampling_rate, preW, sig_prefix, roi, roi_indices)
            
#             # 2. Peak Magnitude
#             ax_mag = fig.add_subplot(gs[row_offset:row_offset+3, col_start+1])
#             _plot_mag_points(ax_mag, peak_results, psth_key, roi, sig_prefix, roi_indices)

#             # 3 & 4. Latency Average & Histograms
#             lats = ['cs_lat', 'rew_lat', 'lick_lat'] if is_rewarded else ['cs_lat']
#             l_colors = [colors['cs'], colors['rew'], colors['lick']]
            
#             for i, (l_type, l_color) in enumerate(zip(lats, l_colors)):
#                 lat_key = f"{psth_key}_peak_{l_type}"
#                 raw_data = peak_results.get(lat_key)
#                 if raw_data is not None:
#                     # Slice for ROI or average across specified Roi2Vis indices
#                     if roi == 'combined':
#                         vals = np.nanmean(raw_data[roi_indices, :], axis=0)
#                     else:
#                         vals = raw_data[roi, :]
                    
#                     # AVG PLOT
#                     ax_avg = fig.add_subplot(gs[row_offset + i, col_start + 2])
#                     _plot_latency_avg(ax_avg, vals, l_color, i == len(lats)-1)
                    
#                     # HIST PLOT
#                     ax_hist = fig.add_subplot(gs[row_offset + i, col_start + 3])
#                     _plot_latency_hist(ax_hist, vals, l_color, i == len(lats)-1)

#     _add_reaction_time_column(fig, gs, TSdict_CSrewarded, cs_name, total_grid_rows)
#     plt.tight_layout()
#     return fig

# # --- Helper Functions (Updated with Trial-Averaging Logic) ---

# def _plot_psth_on_ax(ax, psth_dict, p_key, c_data, fs, preW, sig, roi, roi_indices):
#     data = psth_dict[p_key]
#     time_x = (np.arange(data.shape[0]) - preW) / fs
    
#     if roi == 'combined':
#         # Average only across the specific ROIs in Roi2Vis
#         trial_avg = np.nanmean(data[:, roi_indices, :], axis=1)
#         mean_trace = np.nanmean(trial_avg, axis=1)
#     else:
#         mean_trace = np.nanmean(data[:, roi, :], axis=1)
        
#     ax.plot(time_x, mean_trace, color='green' if sig == 'G' else 'red', lw=2)
#     if c_data is not None:
#         c_mean = np.nanmean(c_data[:, roi_indices if roi=='combined' else roi, :], axis=(1,2) if roi=='combined' else 1)
#         ax.plot(time_x, c_mean, color='gray', alpha=0.5)
    
#     ax.axvline(0, color='black', alpha=0.7)
#     ax.set_title(f"{sig} {'Pooled' if roi=='combined' else f'ROI {roi}'}")

# def _plot_mag_points(ax, results, key, roi, sig, roi_indices):
#     data = results[f"{key}_peak_mag"]
#     vals = np.nanmean(data[roi_indices, :], axis=0) if roi == 'combined' else data[roi, :]
#     vals = vals[~np.isnan(vals)]
    
#     if vals.size > 0:
#         sns.stripplot(y=vals, ax=ax, color='green' if sig == 'G' else 'red', alpha=0.4, jitter=True)
#         ax.errorbar(0, np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)), fmt='ko')
#     ax.set_title("Peak Mag")
#     ax.set_xticks([])

# def _plot_latency_avg(ax, vals, color, is_last):
#     vals = vals[~np.isnan(vals)]
#     if vals.size > 0:
#         ax.scatter(vals, np.random.normal(1, 0.04, len(vals)), color=color, alpha=0.4, s=12)
#         ax.errorbar(np.mean(vals), 1, xerr=np.std(vals)/np.sqrt(len(vals)), fmt='|k', markersize=10)
#         ax.plot(np.mean(vals), 1, 'ko', markersize=4)
#     ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
#     ax.set_xlim([-2, 5]); ax.set_yticks([])
#     if not is_last: ax.set_xticklabels([])

# def _plot_latency_hist(ax, vals, color, is_last):
#     vals = vals[~np.isnan(vals)]
#     if vals.size > 0:
#         sns.histplot(vals, ax=ax, color=color, kde=True, element="step", alpha=0.3, linewidth=0)
#     ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
#     ax.set_xlim([-2, 5]); ax.set_ylabel("")
#     if not is_last: ax.set_xticklabels([])

# def _add_reaction_time_column(fig, gs, beh, cs, total_rows):
#     rt = beh.get(f"{cs}ReactionTime", np.array([]))
#     if rt.size == 0: return
#     ax1 = fig.add_subplot(gs[0:int(total_rows/3), 8])
#     ax1.plot(rt, 'k-o', alpha=0.5, markersize=3); ax1.set_title("RT/Trial")
#     ax2 = fig.add_subplot(gs[int(total_rows/3):int(2*total_rows/3), 8])
#     sns.histplot(rt, ax=ax2, color='black', alpha=0.2, kde=True); ax2.set_title("RT Dist")
#     ax3 = fig.add_subplot(gs[int(2*total_rows/3):total_rows, 8])
#     sns.histplot(rt[rt<0.5], ax=ax3, color='red', alpha=0.3, kde=True); ax3.set_title("Fast RT")



#%% generate_all_trial_summaries
def generate_all_trial_summaries(psth_data, peak_results, TSdict_CSrewarded, 
                                 roi_indices, sampling_rate, preW, save_dir, subjectID, StimPeriod):
    """Loops through all trial types and saves a comprehensive summary figure."""
    trial_types = ['CS1R', 'CS1UR', 'CS2R', 'CS2UR', 'CS3R', 'CS3UR']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for tt in trial_types:
        green_key, red_key = f"G_{tt}_base", f"R_{tt}_base"
        if (green_key in psth_data and psth_data[green_key] is not None) or \
           (red_key in psth_data and psth_data[red_key] is not None):
            try:
                fig = plot_comprehensive_trial_summary(psth_data, peak_results, TSdict_CSrewarded, 
                                                       tt, roi_indices, sampling_rate, preW, subjectID, StimPeriod)
                save_filename = f"{subjectID}_Summary_{tt}_{timestamp}.pdf"
                fig.savefig(os.path.join(save_dir, save_filename), bbox_inches='tight', format='pdf')
            except Exception as e:
                print(f"Error on {tt}: {e}")

def plot_comprehensive_trial_summary(psth_data, peak_results, TSdict_CSrewarded, 
                                     trial_type, roi_indices, sampling_rate, preW, subjectID, StimPeriod):
    is_rewarded = 'R' in trial_type and 'UR' not in trial_type
    n_rois = len(roi_indices)
    total_grid_rows = (n_rois + 1) * 3
    fig = plt.figure(figsize=(30, 4 * (n_rois + 1)))
    gs = gridspec.GridSpec(total_grid_rows, 9, figure=fig)
    fig.suptitle(f"Subject: {subjectID} | Trial Type: {trial_type}", fontsize=22, fontweight='bold', y=1.02)

    plot_list = roi_indices + ['combined']
    for r_idx, roi in enumerate(plot_list):
        row_offset = r_idx * 3
        
        # Calculate Shared Y-Limits for this ROI row ---
        y_min, y_max = 0, 0
        valid_signals = []
        
        for sig_prefix in ['G', 'R']:
            p_key = f"{sig_prefix}_{trial_type}_base"
            if p_key in psth_data and psth_data[p_key] is not None:
                # Extract the data we are actually going to plot
                if roi == 'combined':
                    trace = np.nanmean(psth_data[p_key][:, roi_indices, :], axis=(1, 2))
                else:
                    trace = np.nanmean(psth_data[p_key][:, roi, :], axis=1)
                
                # Update bounds with a small buffer (e.g., 10%)
                y_min = min(y_min, np.nanmin(trace))
                y_max = max(y_max, np.nanmax(trace))
                valid_signals.append(sig_prefix)
        
        # Add 10% padding so the traces don't touch the top/bottom box
        padding = (y_max - y_min) * 0.1
        shared_ylim = (y_min - padding, y_max + padding)
        
        
        for sig_idx, sig_prefix in enumerate(['G', 'R']):
            col_start = sig_idx * 4
            p_key = f"{sig_prefix}_{trial_type}_base"
            c_key = f"C_{trial_type}_base"
            if p_key not in psth_data or psth_data[p_key] is None: continue

            # 1. PSTH (Now using your format)
            ax_psth = fig.add_subplot(gs[row_offset:row_offset+3, col_start])
            _plot_psth_formatted(ax_psth, psth_data, p_key, c_key, sampling_rate, preW,
                                 sig_prefix, roi, roi_indices, StimPeriod, trial_type)

            # Apply the shared limits
            ax_psth.set_ylim(shared_ylim)


            # 2. Peak Mag (Now with (∆F/F) and sig+roi labels)
            ax_mag = fig.add_subplot(gs[row_offset:row_offset+3, col_start+1])
            _plot_mag_points_labeled(ax_mag, peak_results, p_key, roi, sig_prefix, roi_indices)

            # 3 & 4. Latency (Shared Y-labels for Scatter and Hist)
            lats = ['cs_lat', 'rew_lat', 'lick_lat'] if is_rewarded else ['cs_lat']
            y_labels = ['CS', 'Reward', 'Lick']
            l_colors = [[1,0,1,0.4] if 'CS3' in trial_type else [0,1,0,0.4] if 'CS2' in trial_type else [1,0,0,0.4], [0,0,1,0.4], [0,0,0,1]]
            
            for i, (l_type, l_label, l_color) in enumerate(zip(lats, y_labels, l_colors)):
                lat_key = f"{p_key}_peak_{l_type}"
                if lat_key in peak_results:
                    vals = np.nanmean(peak_results[lat_key][roi_indices, :], axis=0) if roi == 'combined' else peak_results[lat_key][roi, :]
                    
                    ax_avg = fig.add_subplot(gs[row_offset + i, col_start + 2])
                    _plot_latency_avg_labeled(ax_avg, vals, l_color, i, l_label)
                    
                    ax_hist = fig.add_subplot(gs[row_offset + i, col_start + 3])
                    _plot_latency_hist_sync(ax_hist, vals, l_color, i == len(lats)-1)

    _add_reaction_time_column(fig, gs, TSdict_CSrewarded, trial_type[:3], total_grid_rows)
    plt.tight_layout()
    return fig

# --- Refined Helpers ---

def _plot_psth_formatted(ax, psth_dict, p_key, c_key, fs, preW, sig, roi, roi_indices, StimPeriod, trial_type):
    # Combined vs Single ROI extraction
    if roi == 'combined':
        psth_subset = np.nanmean(psth_dict[p_key][:, roi_indices, :], axis=1)
        ctrl_subset = np.nanmean(psth_dict[c_key][:, roi_indices, :], axis=1) if c_key in psth_dict else None
    else:
        psth_subset = psth_dict[p_key][:, roi, :]
        ctrl_subset = psth_dict[c_key][:, roi, :] if c_key in psth_dict else None

    # Plot Main Signal
    main_c = 'green' if sig == 'G' else 'red'
    sub_c = [0, 0.8, 0, 0.3] if sig == 'G' else [0.8, 0, 0, 0.3]
    # Reusing your PSTHplot_single_roi logic here
    time_x = np.arange(psth_subset.shape[0]) / fs - preW / fs
    mean_trace = np.nanmean(psth_subset, axis=1)
    sem = np.nanstd(psth_subset, axis=1) / np.sqrt(psth_subset.shape[1])
    
    # CS/Reward Spans
    cs_colors = {'CS1': [1,0,0,0.2], 'CS2': [0,1,0,0.2], 'CS3': [1,0,1,0.2]}
    ax.axvspan(0, 1.0, color=cs_colors.get(trial_type[:3], [0.5,0.5,0.5,0.2]), lw=0)
    if 'R' in trial_type and 'UR' not in trial_type:
        ax.axvspan(2.0, 2.0 + StimPeriod, color=[0,0,1,0.2], lw=0)

    ax.plot(time_x, mean_trace, color=main_c, lw=2, label=f"{sig} (n={psth_subset.shape[1]})")
    ax.fill_between(time_x, mean_trace-sem, mean_trace+sem, color=sub_c, alpha=0.3)
    
    # Control Signal
    if ctrl_subset is not None:
        c_mean = np.nanmean(ctrl_subset, axis=1)
        c_sem = np.nanstd(ctrl_subset, axis=1) / np.sqrt(ctrl_subset.shape[1])
        ax.plot(time_x, c_mean, color=[0, 0, 1, 1.0], lw=1, label='Ctrl')
        ax.fill_between(time_x, c_mean - c_sem, c_mean + c_sem, color=[0, 0, 1], alpha=0.1)
        
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.2)
    ax.legend(loc='upper right', fontsize='xx-small')
    ax.set_title(f"PSTH {'Pooled' if roi=='combined' else f'ROI {roi}'}")

def _plot_mag_points_labeled(ax, results, p_key, roi, sig, roi_indices):
    data = results[f"{p_key}_peak_mag"]
    vals = np.nanmean(data[roi_indices, :], axis=0) if roi == 'combined' else data[roi, :]
    vals = vals[~np.isnan(vals)]
    if vals.size > 0:
        sns.stripplot(y=vals, ax=ax, color='green' if sig == 'G' else 'red', alpha=0.4, jitter=True)
        ax.errorbar(0, np.mean(vals), yerr=np.nanstd(vals)/np.sqrt(len(vals)), fmt='ko')
    ax.set_title(f"Peak Mag (∆F/F)\n{sig} ROI {roi}", fontsize=9)
    ax.set_xticks([])

def _plot_latency_avg_labeled(ax, vals, color, index, l_label):
    if index == 0: ax.set_title("Peak Latency (Relative to Event)", fontsize=9)
    vals = vals[~np.isnan(vals)]
    if vals.size > 0:
        ax.scatter(vals, np.random.normal(1, 0.04, len(vals)), color=color, alpha=0.4, s=12)
        ax.errorbar(np.nanmean(vals), 1, xerr=np.nanstd(vals)/np.sqrt(len(vals)), fmt='|k', markersize=10)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel(l_label, fontsize=8, fontweight='bold')
    ax.set_xlim([-2, 5]); ax.set_yticks([])

def _plot_latency_hist_sync(ax, vals, color, is_last):
    vals = vals[~np.isnan(vals)]
    if vals.size > 0:
        sns.histplot(vals, ax=ax, color=color, kde=True, element="step", alpha=0.3, linewidth=0)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim([-2, 5]); ax.set_ylabel("")
    if not is_last: ax.set_xticklabels([])

def _add_reaction_time_column(fig, gs, beh, cs, total_rows):
    rt_data = np.array(beh.get(f"{cs}ReactionTime", []))
    rt_data = rt_data/1000
    
    if rt_data.size == 0: return

    # Top: Line plot (no markers)
    ax1 = fig.add_subplot(gs[0:int(total_rows/3), 8])
    ax1.plot(rt_data, color='black', lw=1.5)
    ax1.set_title(f"{cs} RT/Trial")

    # Middle: Dist
    ax2 = fig.add_subplot(gs[int(total_rows/3):int(2*total_rows/3), 8])
    sns.histplot(rt_data, ax=ax2, color='black', alpha=0.2, kde=True)
    ax2.set_title("RT Distribution")

    # Bottom: Fast Dist (Fixed filter)
    ax3 = fig.add_subplot(gs[int(2*total_rows/3):total_rows, 8])
    fast_rt = rt_data[rt_data < 0.5]
    if fast_rt.size > 0:
        sns.histplot(fast_rt, ax=ax3, color='red', alpha=0.4, kde=True)
    ax3.set_title(f'Fast RTs (n={len(fast_rt)})')

#%% plot_global_moment_scatter
def plot_global_moment_scatter(g_data, r_data, roi, sid):
    """
    g_psths: 1D array [Time x ROI] 
    r_psths: 2D array [Time x ROI]
    """
    # 1. Flatten all trials and all timepoints into two long vectors
    g_all = g_data.flatten()
    r_all = r_data.flatten()
    
    # 2. Remove any NaNs
    mask = ~np.isnan(g_all) & ~np.isnan(r_all)
    g_clean = g_all[mask]
    r_clean = r_all[mask]

    # 3. Plotting
    plt.figure(figsize=(6, 6))
    
    # Use 'hexbin' or a very high-alpha scatter to see density
    # Hexbin is better for "all trials" because millions of points overlap
    plt.hexbin(r_clean, g_clean, gridsize=50, cmap='viridis', mincnt=1)
    
    x_left, x_right = plt.xlim()
    y_left, y_right = plt.ylim()
    
    # Add a unity line (y-x)
    # Find the overall limits to make the line long enough
    min_limit = min(np.min(g_clean), np.min(r_clean))
    max_limit = max(np.max(g_clean), np.max(r_clean))
    plt.plot([min_limit, max_limit], [min_limit, max_limit], color='black', 
             linestyle=':', alpha=0.6, label='(Y=X)')
    
    # Add the linear regression line
    slope, intercept, r_val, p_val, _ = stats.linregress(r_clean, g_clean)
    x_range = np.array([np.min(r_clean), np.max(r_clean)])
    plt.plot(x_range, slope*x_range + intercept, color='red', linestyle='--', 
             label=f'Global R²={r_val**2:.3f}')

    # set axes limits back to original
    plt.xlim(x_left, x_right)
    plt.ylim(y_left, y_right)
    
    plt.xlabel('Red Signal')
    plt.ylabel('Green Signal')
    plt.title(f'Moment-to-Moment Coupling:{sid} ROI {roi}\n(All Timepoints)')
    plt.colorbar(label='Point Density')
    plt.legend()
    sns.despine()
    plt.show()



#%% plot_peak_coupling
def plot_peak_coupling(peak_data, second_data, roi_idx, threshold, sid, fs, peak_channel_name):
    """
    Finds peaks in peak_data and plots their amplitudes against 
    the values in second_data at the same timestamps.
    """
    # 1. Extract the specific ROI column [Time]
    ref_signal = peak_data[:, roi_idx]
    other_signal = second_data[:, roi_idx]

    # 2. Find peaks above threshold
    # threshold is in % dF/F as per your data normalization, minimum 2s apart
    peak_indices, _ = find_peaks(ref_signal, height=threshold, distance=fs*0.5)

    if len(peak_indices) == 0:
        print(f"No peaks found above {threshold}% in ROI {roi_idx}")
        return

    # 3. Get the amplitudes at those indices
    x = ref_signal[peak_indices]  # Amplitudes of the peak channel
    y = other_signal[peak_indices] # Corresponding values in second channel

    # 4. Regression Analysis
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    fig, ax1 = plt.subplots(figsize=(5, 5))

    if len(x_clean) > 1:
        slope, intercept, r_val, p_val, _ = stats.linregress(x_clean, y_clean)
        
        # Plot raw points
        ax1.scatter(x_clean, y_clean, color='gray', alpha=0.3, s=25, edgecolors='none')
        
        # Plot regression line
        line = slope * x_clean + intercept
        ax1.plot(x_clean, line, color='red', lw=2, label=f'$R^2$={r_val**2:.3f}\n$p$={p_val:.4e}')
        
        ax1.set_title(f'Amplitude Coupling: {sid} for {peak_channel_name} Peaks\n(ROI {roi_idx})')
        ax1.set_xlabel('Peak Channel (% $\Delta F/F$)')
        ax1.set_ylabel('Second Channel (% $\Delta F/F$)')
        ax1.legend(frameon=False)
        plt.tight_layout()
        
    return fig, x_clean, y_clean

#%% analyze_peak_coupling
def analyze_peak_coupling(data_primary, data_secondary, time_seconds, roi_idx, 
                          threshold, fs, sid, peak_channel_name, window_sec=[2, 2]):
    """
    Unified function to plot amplitude coupling and sample individual transients.
    Works for any channel passed as 'data_primary'.
    """
    # Determine colors and names based on input
    if peak_channel_name.lower() == 'green':
        c_prim, c_sec = 'green', 'magenta'
        other_name = 'Red'
    else:
        c_prim, c_sec = 'magenta', 'green'
        other_name = 'Green'

    # 1. Extract Signals
    sig_prim = data_primary[:, roi_idx]
    sig_sec = data_secondary[:, roi_idx]
    
    # if signal is for more than one ROI, flatten:
    if sig_prim.ndim > 1:
        sig_prim = sig_prim.flatten()
        sig_sec = sig_sec.flatten()
        n_trials = data_primary.shape[1]
        time_seconds = np.tile(time_seconds, n_trials)
        
    # 2. Find ALL peaks in primary channel
    p_idx, _ = find_peaks(sig_prim, height=threshold, distance=fs*2)
    
    if len(p_idx) < 2:
        print(f"Insufficient peaks found in ROI {roi_idx}")
        return None, None

    # 3. Targeted Selection: Find peaks where Secondary Signal is <= 0
    sec_values_at_peaks = sig_sec[p_idx]
    
    # # Indices within the p_idx array where sec signal is low/negative
    # low_sec_indices = np.where(sec_values_at_peaks <= 0)[0]
    
    # if len(low_sec_indices) >= 10:
    #     # If we have plenty, take a random sample of 10 from the "low" group
    #     highlight_indices = random.sample(list(low_sec_indices), 10)
    # elif len(low_sec_indices) > 0:
    #     # If we have some but fewer than 10, take all of them
    #     highlight_indices = list(low_sec_indices)
    # else:
    #     # LAST RESORT: Take the 10 lowest values available
    #     print(f"Note: No peaks found with {other_name} <= 0. Selecting 10 lowest values.")
    #     highlight_indices = np.argsort(sec_values_at_peaks)[:10]
   
    highlight_indices = np.argsort(sec_values_at_peaks)[:10]
    
    highlight_p_idx = p_idx[highlight_indices]
    

    # --- FIGURE 1: Amplitude Coupling ---
    fig1, ax_corr = plt.subplots(figsize=(6, 6))
    
    x_all, y_all = sig_prim[p_idx], sig_sec[p_idx]
    
    ax_corr.scatter(x_all, y_all, color='gray', alpha=0.3, s=30, label='All Peaks', edgecolors='none')
    ax_corr.scatter(x_all[highlight_indices], y_all[highlight_indices], 
                    color='red', s=60, edgecolors='black', label='Highlighted Samples', zorder=5)
    
    # Regression
    slope, intercept, r_val, p_val, _ = stats.linregress(x_all, y_all)
    ax_corr.plot(x_all, slope*x_all + intercept, color='black', linestyle='--', alpha=0.7)
    
    ax_corr.set_title(f'{sid} (ROI {roi_idx})\nRef: {peak_channel_name.capitalize()} | $R^2$={r_val**2:.3f}')
    ax_corr.set_xlabel(f'{peak_channel_name.capitalize()} Peak Amplitude (% $\Delta F/F$)')
    ax_corr.set_ylabel(f'{other_name} Value at Peak (% $\Delta F/F$)')
    ax_corr.legend(frameon=False)
    sns.despine()

    # --- FIGURE 2: Individual Trace Windows ---
    fig2, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    pre_s, post_s = int(window_sec[0] * fs), int(window_sec[1] * fs)
    
    for i, peak_time_idx in enumerate(highlight_p_idx):
        start = max(0, peak_time_idx - pre_s)
        end = min(len(sig_prim), peak_time_idx + post_s)
        rel_time = (np.arange(start, end) - peak_time_idx) / fs
        
        # Plot traces
        axes[i].plot(rel_time, sig_prim[start:end], color=c_prim, lw=2, label=peak_channel_name)
        
        # Calculate offset for visualization
        raw_offset = np.nanmax(sig_prim[start:end]) - np.nanmin(sig_sec[start:end]) + 5
        # Round to nearest 5
        offset = 5 * round(raw_offset / 5)
        
        axes[i].plot(rel_time, sig_sec[start:end] - offset, color=c_sec, lw=2, label=other_name)
        
        axes[i].axvline(0, color='red', linestyle=':', alpha=0.6)
        axes[i].set_title(f"Peak: {time_seconds[peak_time_idx]:.1f}s")
        
        if i >= 5: axes[i].set_xlabel('Time (s)')
        if i % 5 == 0: axes[i].set_ylabel('% $\Delta F/F$')

    plt.suptitle(f"Samples (Ref: {peak_channel_name}) - {sid} (ROI {roi_idx}. \nThreshold: {threshold}    Offset: {offset})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig1, fig2, highlight_indices, highlight_p_idx

#%% save_analysis_to_hdf5
def save_analysis_to_hdf5(save_dir, subjectID, psth_data, psth_pooled_data, 
                          rt_data, peak_results, TSdict, TSdict_CSrewarded, 
                          Roi2Vis, sampling_rate, preW, StimPeriod):
    """
    Saves all photometry analysis data and metadata into a single HDF5 file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{subjectID}_preprocessed_{timestamp}.h5"
    h5_filename = os.path.join(save_dir, file_name)
    
    print(f"Saving analysis data to {h5_filename}...")

    with h5py.File(h5_filename, 'w') as hf:
        # 1. Metadata / Attributes
        hf.attrs['subjectID'] = subjectID
        hf.attrs['sampling_rate'] = sampling_rate
        hf.attrs['preW'] = preW
        hf.attrs['StimPeriod'] = StimPeriod
        hf.create_dataset('Roi2Vis', data=np.array(Roi2Vis))

        # 2. PSTH Data (Groups for individual and pooled)
        g_psth = hf.create_group('psth_data')
        for key, val in psth_data.items():
            if val is not None:
                g_psth.create_dataset(key, data=val, compression="gzip")

        g_pooled = hf.create_group('psth_pooled_data')
        for key, val in psth_pooled_data.items():
            if val is not None:
                g_pooled.create_dataset(key, data=val, compression="gzip")

        # 3. Peak Results (Group)
        g_peaks = hf.create_group('peak_results')
        for key, val in peak_results.items():
            if val is not None:
                g_peaks.create_dataset(key, data=val)

        # 4. Reaction Time Data 
        if rt_data is not None:
            hf.create_dataset('rt_data', data=np.array(rt_data))

        # 5. Timestamps (Groups for TSdict and TSdict_CSrewarded)
        g_ts = hf.create_group('TSdict')
        for key, val in TSdict.items():
            if val is not None:
                g_ts.create_dataset(key, data=np.array(val))

        g_ts_rew = hf.create_group('TSdict_CSrewarded')
        for key, val in TSdict_CSrewarded.items():
            if val is not None:
                g_ts_rew.create_dataset(key, data=np.array(val))

    print("Successfully saved all data to HDF5.")
    return h5_filename