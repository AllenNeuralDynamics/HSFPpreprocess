# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:30:40 2026

Simplified FIP preprocessing/plotting

TrialType_
1:CS1 Rewarded
2-10:CS1 UnRewarded
11-15:CS2 Rewarded
16-20CS2 UnRewarded
21-29:CS3 Rewarded
30: CS3 UnRewarded

@author: carrie.stine
"""
#%% setup/imports
import os
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import numpy as np
# import csv
# import glob
# import re
# from scipy.optimize import curve_fit
# import json
# import pandas as pd
# from scipy.stats import sem
import h5py

#import PreprocessingFunctions2 as pf
import FIPFunctions2 as fipf


session_id = 'FIP_835527_2026-01-12_10-31-26'

SaveDir = r'C:\output_data\results\results_' + session_id
AnalDir = r'C:\output_data' + os.sep + session_id + os.sep + 'behavior'
FibDir = r'C:\output_data' + os.sep + session_id + os.sep + 'fib'
os.makedirs(SaveDir, exist_ok=True)

# manually enter when the first reward trial that should actually be counted happened
first_rew_idx = 0           # 0 for 836733 12/3/25
                            # 23 for 836732 12/3/25
# choose which ROIs (fibers) to visualize
#Roi2Vis=[0,1,2]
Roi2Vis = [0,1]

SaveFigs = 0 # set to 1 to save generated plots in results directory, 0 to skip
SaveResults = 0 # set to 1 to save preprocessed_results in fib directory, 0 to skip

# params for pre-processing
nFrame2cut = 100  #crop initial n frames
sampling_rate = 20 #individual channel (not total)
kernelSize = 1 #median filter
degree = 4 #polyfit
b_percentile = 0.70 #To calculate F0, median of bottom x%

StimPeriod = 0.1 #sec for visualization, used to show time duration of rew delivery`
preW = 100 #nframes for PSTH (before event)
postW = 300 #nframes for PSTH (after event)
LickWindow = 5.0 #sec window length for Consummatory/Omission licks
PeakWindow = [0, 5.0] # sec window length to search for peaks, relative to CS onset
#%% Load the data
# format for data1/2/3 = [m, n] array where m (rows) = number of timestamps and  
# n (cols) = number of ROIs + 2 (first column = SoftwareTS, last column = HarpTS)

# data1 = isos, data2 = green, data3 = red
data1, data2, data3, subjectID, TSdict = fipf.load_fip_data(AnalDir)


#%% Trim out initial manual rewards
TSdict['Reward'] = TSdict['Reward'][first_rew_idx:, :]


#%% Sync lengths and get session time
data1, data2, data3, PMts, time_seconds = fipf.sync_and_time(data1, data2, data3, sampling_rate)


#%% Preprocess the data
# preprocessing skips the first column in data1/2/3 since this is just SoftwareTS
# output format for X_dF_F = [m, n]  array where m (rows) = number of timestamps and  
# n (cols) = number of ROIs + 1 (last column = HarpTS)
Ctrl_dF_F, G_dF_F, R_dF_F = fipf.preprocess_all_channels(
    data1, data2, data3, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile
)


#%% Extract event timestamps into frame indices
event_frames = fipf.get_event_frames(TSdict, data1[:, 0])
RewardFrames = event_frames.get('Reward', [])
LickFrames   = event_frames.get('Lick', [])
CS1Frames   = event_frames.get('CS1', [])
CS2Frames   = event_frames.get('CS2', [])
CS3Frames   = event_frames.get('CS3', [])
ManualRewardFrames = event_frames.get('ManualReward', [])


#%% Load pupil data (optional)
pupil_time, pupil_data = fipf.load_pupil_data(AnalDir, data1[0, 0])


#%% Plot the entire trace
# Pack event traces for plotting
events = {
    'Reward': RewardFrames,
    'CS1': CS1Frames,
    'CS2': CS2Frames,
    'CS3': CS3Frames,
    'Lick': LickFrames
}

# Pack pupil data for plotting (if it exists)
pupil_input = (pupil_time, pupil_data) if 'pupil_data' in locals() and pupil_data is not None else None

fig_wholetrace = fipf.plot_whole_trace(
    time_seconds, Ctrl_dF_F, G_dF_F, R_dF_F, Roi2Vis, 
    events, subjectID, AnalDir, StimPeriod, 
    ds_factor=1, pupil_data=pupil_input
    )   

if SaveFigs == 1:
    fig_wholetrace.savefig(os.path.join(SaveDir, f"{subjectID}_WholeTrace_Summary.pdf"), bbox_inches='tight')

#%% Plot a short window of the full trace
# 1. Define your window in indices (20Hz)
zoom_start = 150 # time in seconds
zoom_end = 350 

# Convert to indices
idx_start = zoom_start * sampling_rate
idx_end = zoom_end * sampling_rate    

# 2. Slice the main data arrays
time_subset = time_seconds[idx_start:idx_end]
Ctrl_subset = Ctrl_dF_F[idx_start:idx_end, :]
G_subset    = G_dF_F[idx_start:idx_end, :]
R_subset    = R_dF_F[idx_start:idx_end, :]

# 3. Filter the events
events_subset = {}
for key, frames in events.items():
    subset = [f for f in frames if idx_start <= f < idx_end]
    events_subset[key] = np.array(subset)

# 4. Filter Pupil Data (Keep original time)
if pupil_input is not None:
    p_time, p_vals = pupil_input
    p_mask = (p_time >= zoom_start) & (p_time <= zoom_end)
    pupil_subset = (p_time[p_mask], p_vals[p_mask])
else:
    pupil_subset = None

# 5. Call the Function
fig_zoom = fipf.plot_whole_trace(
    time_subset, Ctrl_subset, G_subset, R_subset, Roi2Vis, 
    events_subset, subjectID, AnalDir, StimPeriod,
    ds_factor=1, pupil_data=pupil_subset
    )

# 6. Adjust X-axis limits to focus specifically on that window
plt.xlim([zoom_start, zoom_end])
plt.title(f"Subject: {subjectID} | Actual Time: {zoom_start}s - {zoom_end}s")
if SaveFigs == 1:
    fig_zoom.savefig(os.path.join(SaveDir, f"{subjectID}_ZoomTrace_Summary.pdf"), bbox_inches='tight')

#%% Extract trial types (rewarded vs unrewarded)
# format of trial_data = dict of size 9 with following 3 entry types for CS1, CS2, and CS3:
    # 1: Mat_CSX = indices for all CSX trials
    # 2: RewardedCSXind = indices for all rewarded CSX trials
    # 3: UnRewardedCSXind = indices for all unrewarded CSX trials

# NOTE: will only get trial types R/UR for CS1/CS2/CS3, will need to update the function to get any other types
trial_data = fipf.get_trial_indices(AnalDir)

# Store indices for rewarded and unrewarded trials of each CS type (optional)
R_idx_CS1 = trial_data['RewardedCS1ind']
UR_idx_CS1 = trial_data['UnRewardedCS1ind']
R_idx_CS2 = trial_data['RewardedCS2ind']
UR_idx_CS2 = trial_data['UnRewardedCS2ind']
R_idx_CS3 = trial_data['RewardedCS3ind']
UR_idx_CS3 = trial_data['UnRewardedCS3ind']

TrialFrames = np.sort(np.concatenate([CS1Frames, CS2Frames, CS3Frames]))
trial_times =time_seconds[TrialFrames.astype(int)]
trial_iti = np.diff(trial_times)
min_iti = np.min(trial_iti)
max_iti = np.max(trial_iti)
print(f'Minimum ITI:{min_iti} s')
print(f'Maximum ITI:{max_iti} s')

#%% Calculate PSTH for signal around all trials
# format of psth_data = dict of size 18 for each combination of the following:
    # 1: (3) CS_types (CS1, CS2, CS3)
    # 2: (2) trial_types (rewarded, unrewarded)
    # 3: (3) data_types (C = control/isos, G = green, R = red)

# example for accessing psth_data: Green CS3 rewarded = psth_data['G_CS3R_base']
psth_data = fipf.generate_all_psths(G_dF_F, R_dF_F, Ctrl_dF_F, event_frames, trial_data, preW, postW)


# FORMAT - each entry in psth_data has size (x, y, z) where:
    # x = Time: preW + postW (standard = 400)
    # y = ROI: each fiber that was recorded from (typically only using 0 and 1)
    # z = Trials: # of individual trials for the given type (e.g. CS3 rewarded)

#%% Pool data from each ROI together (ONLY IF RECORDING SITES ARE THE SAME)
psth_pooled_data = fipf.pool_rois_in_psths(psth_data, Roi2Vis)


#%% Create a dictionary of timestamps for rewards and licks by CS type
TSdict_CSrewarded = fipf.sort_timestamps_by_cs(TSdict, trial_data)

#%% Calculate photometry peaks relative to CS onset, reward delivery, and first lick
peak_results = fipf.calculate_fip_peaks(psth_data, TSdict_CSrewarded, sampling_rate, preW, PeakWindow)

#%%
fipf.plot_latency_comparison(peak_results, trial_type='CS3R', roi_idx=0)


#%% Plot PSTHs for each trial type with separate ROIs
# Plot CS1
# CS1 Rewarded
if psth_data.get('G_CS1R_base') is not None:
    fig_psth_CS1R = fipf.plot_roi_psth_summary(psth_data, subjectID, Roi2Vis, sampling_rate, StimPeriod, preW, 
                                               trial_type='CS1R')
    fig_peaks_CS1R = fipf.plot_time_to_peak_summary(psth_data, subjectID, Roi2Vis, sampling_rate, preW, 
                                                    trial_type='CS1R', search_window=PeakWindow)
    if SaveFigs == 1:
        fig_psth_CS1R.savefig(os.path.join(SaveDir, f"{subjectID}_CS1R_PSTH_ROI-Summary.svg"), bbox_inches='tight')
        fig_peaks_CS1R.savefig(os.path.join(SaveDir, f"{subjectID}_CS1R_peaks_ROI-Summary.svg"), bbox_inches='tight')
else:
    print("Skipping CS1R: No trials found.")

# CS1 Unrewarded
if psth_data.get('G_CS1UR_base') is not None:
    fig_psth_CS1UR = fipf.plot_roi_psth_summary(psth_data, subjectID, Roi2Vis, sampling_rate, StimPeriod, preW, 
                                           trial_type='CS1UR')
    fig_peaks_CS1UR = fipf.plot_time_to_peak_summary(psth_data, subjectID, Roi2Vis, sampling_rate, preW, 
                                                    trial_type='CS1UR', search_window=PeakWindow)
    if SaveFigs == 1:
        fig_psth_CS1UR.savefig(os.path.join(SaveDir, f"{subjectID}_CS1UR_PSTH_ROI-Summary.svg"), bbox_inches='tight')
        fig_peaks_CS1UR.savefig(os.path.join(SaveDir, f"{subjectID}_CS1UR_peaks_ROI-Summary.svg"), bbox_inches='tight')
else:
    print("Skipping CS1UR: No trials found.")
    
    
    
# Plot CS2
# CS2 Rewarded
if psth_data.get('G_CS2R_base') is not None:
    fig_psth_CS2R = fipf.plot_roi_psth_summary(psth_data, subjectID, Roi2Vis, sampling_rate, StimPeriod, preW, 
                                               trial_type='CS2R')
    fig_peaks_CS2R = fipf.plot_time_to_peak_summary(psth_data, subjectID, Roi2Vis, sampling_rate, preW, 
                                                    trial_type='CS2R', search_window=PeakWindow)
    if SaveFigs == 1:
        fig_psth_CS2R.savefig(os.path.join(SaveDir, f"{subjectID}_CS2R_PSTH_ROI-Summary.svg"), bbox_inches='tight')
        fig_peaks_CS2R.savefig(os.path.join(SaveDir, f"{subjectID}_CS2R_peaks_ROI-Summary.svg"), bbox_inches='tight')
else:
    print("Skipping CS2R: No trials found.")
    
# CS2 Unrewarded
if psth_data.get('G_CS2UR_base') is not None:
    fig_psth_CS2UR = fipf.plot_roi_psth_summary(psth_data, subjectID, Roi2Vis, sampling_rate, StimPeriod, preW, 
                                                trial_type='CS2UR')
    fig_peaks_CS2UR = fipf.plot_time_to_peak_summary(psth_data, subjectID, Roi2Vis, sampling_rate, preW,
                                                     trial_type='CS2UR', search_window=PeakWindow)
    if SaveFigs == 1:
        fig_psth_CS2UR.savefig(os.path.join(SaveDir, f"{subjectID}_CS2UR_PSTH_ROI-Summary.svg"), bbox_inches='tight')
        fig_peaks_CS2UR.savefig(os.path.join(SaveDir, f"{subjectID}_CS2UR_peaks_ROI-Summary.svg"), bbox_inches='tight')
else:
    print("Skipping CS2UR: No trials found.")
    

    
    
# Plot CS3
# CS3 Rewarded
if psth_data.get('G_CS3R_base') is not None:
    fig_psth_CS3R = fipf.plot_roi_psth_summary(psth_data, subjectID, Roi2Vis, sampling_rate, StimPeriod, preW, 
                                               trial_type='CS3R')
    fig_peaks_CS3R = fipf.plot_time_to_peak_summary(psth_data, subjectID, Roi2Vis, sampling_rate, preW, 
                                                    trial_type='CS3R', search_window=PeakWindow)
    if SaveFigs == 1:
        fig_psth_CS3R.savefig(os.path.join(SaveDir, f"{subjectID}_CS3R_PSTH_ROI-Summary.svg"), bbox_inches='tight')
        fig_peaks_CS3R.savefig(os.path.join(SaveDir, f"{subjectID}_CS3R_peaks_ROI-Summary.svg"), bbox_inches='tight')
else:
    print("Skipping CS3R: No trials found.")
    
# CS3 Unrewarded
if psth_data.get('G_CS3UR_base') is not None:
    fig_psth_CS3UR = fipf.plot_roi_psth_summary(psth_data, subjectID, Roi2Vis, sampling_rate, StimPeriod, preW, 
                                                trial_type='CS3UR')
    fig_peaks_CS3UR = fipf.plot_time_to_peak_summary(psth_data, subjectID, Roi2Vis, sampling_rate, preW,
                                                     trial_type='CS3UR', search_window=PeakWindow)
    if SaveFigs == 1:
        fig_psth_CS3UR.savefig(os.path.join(SaveDir, f"{subjectID}_CS3UR_PSTH_ROI-Summary.svg"), bbox_inches='tight')
        fig_peaks_CS3UR.savefig(os.path.join(SaveDir, f"{subjectID}_CS3UR_peaks_ROI-Summary.svg"), bbox_inches='tight')
else:
    print("Skipping CS3UR: No trials found.")


#%% Plot reaction time from reward delivery
rt_data = fipf.calculate_and_plot_rt(TSdict, subjectID, SaveDir, save_figs=SaveFigs)


#%% Big summary plot
fipf.generate_all_trial_summaries(
    psth_data, 
    peak_results, 
    TSdict_CSrewarded, 
    Roi2Vis,          # Your list of ROIs to plot
    sampling_rate, 
    preW, 
    SaveDir, 
    subjectID,
    StimPeriod
)

#%% plot moment to moment brightness by ROI
for roi in Roi2Vis:  
    g_data = G_dF_F[:,roi]*100
    r_data = R_dF_F[:,roi]*100
    fipf.plot_global_moment_scatter(g_data, r_data, roi, subjectID)

# Combine ROIs and visualize
roi = 'Combined'
roi_idx = [i for i in Roi2Vis]
g_data = G_dF_F[:,roi_idx]*100
r_data = R_dF_F[:,roi_idx]*100
fipf.plot_global_moment_scatter(g_data, r_data, roi, subjectID)


#%% Find peaks in one channel and plot against amplitude of other channel
g_peak_thresh = 5
r_peak_thresh = 5
G_dF_F_percent = G_dF_F * 100
R_dF_F_percent = R_dF_F * 100

#Z-score data
n_frames = G_dF_F.shape[0]
n_rois = len(Roi2Vis)
G_Z = np.zeros((n_frames, n_rois))
R_Z = np.zeros((n_frames, n_rois))
for roi in Roi2Vis:
    G_Z[:,roi] = (G_dF_F[:, roi] - np.mean(G_dF_F[:, roi])) / (np.std(G_dF_F[:, roi]) + 1e-6)
    R_Z[:,roi] = (R_dF_F[:, roi] - np.mean(R_dF_F[:, roi])) / (np.std(R_dF_F[:, roi]) + 1e-6)

# initialize arrays to hold indices
G_highlighted_indices = []
G_p_idx = []
R_highlighted_indices = []
R_p_idx = []

for roi in roi_idx:
    fig_greenpeaks, fig_greenexamples, highlighted_indices, highlight_p_idx = fipf.analyze_peak_coupling(
        G_dF_F_percent, R_dF_F_percent, # DF/F
        #G_Z, R_Z, # Z-score
        time_seconds, roi, g_peak_thresh, sampling_rate, subjectID, 'green', window_sec = [2, 2]
        )
    
    G_highlighted_indices.append(highlighted_indices)
    G_p_idx.append(highlight_p_idx)
    
    fig_redpeaks, fig_redexamples, highlighted_indices, highlight_p_idx = fipf.analyze_peak_coupling(
        R_dF_F_percent, G_dF_F_percent, # DF/F
        #R_Z, G_Z, # Z-score
        time_seconds, roi, r_peak_thresh, sampling_rate, subjectID, 'red', window_sec = [2, 2]
        )
    
    R_highlighted_indices.append(highlighted_indices)
    R_p_idx.append(highlight_p_idx)
    
    if SaveFigs == 1:
        # Save the green peaks coupled to red signal + examples of uncoupled responses
        fig_greenpeaks.savefig(os.path.join(SaveDir, f"{subjectID}_ROI{roi}_greenpeaks_coupling.svg"), bbox_inches='tight')
        fig_greenexamples.savefig(os.path.join(SaveDir, f"{subjectID}_ROI{roi}_greenpeaks_uncoupled-examples.svg"), bbox_inches='tight')
        
        # Save the red peaks coupled to green signal + examples of uncoupled responses
        fig_redpeaks.savefig(os.path.join(SaveDir, f"{subjectID}_ROI{roi}_redpeaks_coupling.svg"), bbox_inches='tight')
        fig_redexamples.savefig(os.path.join(SaveDir, f"{subjectID}_ROI{roi}_redpeaks_uncoupled-examples.svg"), bbox_inches='tight')
        
        
#%% Save preprocessed results to fib directory as HDF5

if SaveResults == 1:
    preprocessed_results = fipf.save_analysis_to_hdf5(FibDir, subjectID, psth_data, psth_pooled_data, 
                              rt_data, peak_results, TSdict, TSdict_CSrewarded, 
                              Roi2Vis, sampling_rate, preW, StimPeriod)







#%% Test area

# Figure out which highlighted peaks are used in final plot (adjust idx_keep to the subset of the 10 that you want to keep)
highlighted_x = G_dF_F_percent[G_p_idx[1], 1]
highlighted_y = R_dF_F_percent[G_p_idx[1], 1]
idx_keep = [0, 2, 3, 4, 5, 8]
x_keep =highlighted_x[idx_keep] 
y_keep =highlighted_y[idx_keep] 
fig1, ax_corr = plt.subplots(figsize=(6, 6))
ax_corr.scatter(x_keep, y_keep, color='red', alpha=0.9, s=60, label='All Peaks', edgecolors='black')
ax_corr.set_xlim([0, 80])
ax_corr.set_ylim([-6, 20])

#%%
fig_greenpeaks, fig_greenexamples, highlighted_indices, highlight_p_idx = fipf.analyze_peak_coupling(
    G_dF_F_percent, R_dF_F_percent, # DF/F
    #G_Z, R_Z, # Z-score
    time_seconds, Roi2Vis, g_peak_thresh, sampling_rate, subjectID, 'green', window_sec = [2, 2]
    )