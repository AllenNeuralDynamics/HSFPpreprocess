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
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
import glob
import re
import pandas as pd

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

#%% Define PSTH plotting function

# DEPRECATED: Use PSTHplot_single_roi for grid summaries

# def PSTHplot(PSTH, MainColor, SubColor, LabelStr, preW, sampling_rate):
#     plt.plot(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, np.mean(PSTH.T,axis=1),label=LabelStr,color = MainColor)
#     #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
#     #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
#     y11 =  np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
#     y22 =  np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
#     plt.fill_between(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, y11, y22, facecolor=SubColor, alpha=0.5)


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

 #%% plot_roi_psth_summary DEPRECATED
# def plot_roi_psth_summary(psths, Roi2Vis, sampling_rate, StimPeriod, preW=100, trial_type='CS3R'):
#     """
#     Creates a grid where:
#     - Each ROW is a different ROI.
#     - Left Column: Green + Control
#     - Right Column: Red + Control
#     - Y-axes are matched within each ROI (row).
#     """
#     num_rois = len(Roi2Vis)
#     # 2 columns (Green vs Red), and 'num_rois' rows
#     fig, axes = plt.subplots(num_rois, 2, figsize=(10, 4 * num_rois), sharex=True)
    
#     # Handle the case where only 1 ROI is selected (axes becomes 1D)
#     if num_rois == 1:
#         axes = np.expand_dims(axes, axis=0)

#     for i, roi_idx in enumerate(Roi2Vis):
#         # Identify the axes for this row
#         ax_green_col = axes[i, 0]
#         ax_red_col = axes[i, 1]
        
#         # Get data
#         g_data = psths.get(f'G_{trial_type}_base')
#         r_data = psths.get(f'R_{trial_type}_base')
#         c_data = psths.get(f'C_{trial_type}_base')

#         # --- LEFT COLUMN: GREEN + CONTROL ---
#         if g_data is not None and c_data is not None:
#             PSTHplot_single_roi(g_data[:, roi_idx, :], 'green', [0, 0.8, 0], 'Green', preW, sampling_rate, ax_green_col, StimPeriod, trial_type=trial_type)
#             PSTHplot_single_roi(c_data[:, roi_idx, :], 'blue', [0, 0, 0.8], 'Ctrl', preW, sampling_rate, ax_green_col, StimPeriod, trial_type=trial_type)
        
#         ax_green_col.set_title(f'ROI {roi_idx}: Green + Ctrl')
#         ax_green_col.set_ylabel('dF/F (%)')

#         # --- RIGHT COLUMN: RED + CONTROL ---
#         if r_data is not None and c_data is not None:
#             PSTHplot_single_roi(r_data[:, roi_idx, :], 'magenta', [0.8, 0, 0.8], 'Red', preW, sampling_rate, ax_red_col, StimPeriod, trial_type=trial_type)
#             PSTHplot_single_roi(c_data[:, roi_idx, :], 'blue', [0, 0, 0.8], 'Ctrl', preW, sampling_rate, ax_red_col, StimPeriod, trial_type=trial_type)
            
#         ax_red_col.set_title(f'ROI {roi_idx}: Red + Ctrl')
#         ax_red_col.set_ylabel('dF/F (%)')
        
#         # Match Y-axes for this row (this ROI)
#         y1_min, y1_max = ax_green_col.get_ylim()
#         y2_min, y2_max = ax_red_col.get_ylim()
#         common_min = min(y1_min, y2_min)
#         common_max = max(y1_max, y2_max)
        
#         ax_green_col.set_ylim(common_min, common_max)
#         ax_red_col.set_ylim(common_min, common_max)

#     # Only add X-label to the bottom row
#     axes[-1, 0].set_xlabel('Time (s)')
#     axes[-1, 1].set_xlabel('Time (s)')

#     fig.suptitle(f'Summary PSTH: {trial_type}', fontsize=16, fontweight='bold', y=0.98)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     return fig
    
#%%
def plot_roi_psth_summary(psths, Roi2Vis, sampling_rate, StimPeriod, preW=100, trial_type='CS3R'):
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

    fig.suptitle(f'Summary PSTH: {trial_type}', fontsize=16, fontweight='bold', y=0.98)
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
    
    
    
    
# def PSTHplot_single_roi(PSTH_subset, MainColor, SubColor, LabelStr, preW, sampling_rate, ax):
#     """Plots a PSTH subplot for G+Ctrl or R+Ctrl"""
#     time_x = np.arange(PSTH_subset.shape[0]) / sampling_rate - preW / sampling_rate
#     mean_trace = np.mean(PSTH_subset, axis=1)
#     sem = np.std(PSTH_subset, axis=1) / np.sqrt(PSTH_subset.shape[1])
    
#     # 1. Add Shaded Event Spans
#     # CS Span: Starts at 0, lasts for StimPeriod (e.g., 0.5s or 1.0s)
#     ax.axvspan(0, 1.0, color=[1, 0, 1, 0.2], lw=0, label='CS3' if LabelStr=='Green' else "")
    
    
#     ax.plot(time_x, mean_trace, label=LabelStr, color=MainColor, lw=2)
#     ax.fill_between(time_x, mean_trace - sem, mean_trace + sem, facecolor=SubColor, alpha=0.3)
#     ax.axvline(0, color='black', linestyle='--', alpha=0.5) # Stimulus onset
#     ax.legend(loc='upper right', fontsize='small')