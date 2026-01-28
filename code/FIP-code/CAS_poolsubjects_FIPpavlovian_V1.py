# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:55:54 2026

@author: carrie.stine
"""
import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy import signal

import FIPFunctions_processed as fipf_p


# Illustrator Compatibility: Ensure text is saved as editable fonts (Type 42)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#%%
# 1. Define your list of session IDs
session_ids = [
    'FIP_835527_2026-01-12_10-31-26',
    'FIP_840205_2026-01-12_09-58-41',
    # '', # Add more IDs here
]


SaveDir = r'C:\output_data\results\combined'
# 1. Ensure the SaveDir exists before saving
if not os.path.exists(SaveDir):
    os.makedirs(SaveDir)

# 2. Storage for all reloaded data
cohort_data = {}

for session_id in session_ids:
    session_path = os.path.join(r'C:\output_data', session_id, 'fib')
    print(f"Searching in: {session_path}")

    # Search for any .h5 file
    h5_files = glob.glob(os.path.join(session_path, "*.h5"))
    
    if not h5_files:
        if os.path.exists(session_path):
            print(f"Folder exists, but no .h5 found. Folder contains: {os.listdir(session_path)}")
        else:
            print(f"Error: The directory path does not exist: {session_path}")
        continue # This is correct here, as we can't load what isn't there
    
    # 3. If we found a file, proceed to load it
    target_file = h5_files[0]
    print(f"Found file: {os.path.basename(target_file)}")
    
    try:
        # Note: Ensure the function name matches (load_fip_h5 vs load_fip_hdf5)
        loaded_vars = fipf_p.load_fip_h5(target_file)
        
        # Store the results
        cohort_data[session_id] = {
            'psth_data': loaded_vars[0],
            'psth_pooled': loaded_vars[1],
            'rt_data': loaded_vars[2],
            'peak_results': loaded_vars[3],
            'TSdict': loaded_vars[4],
            'TSdict_rew': loaded_vars[5],
            'Roi2Vis': loaded_vars[6],
            'fs': loaded_vars[7],
            'preW': loaded_vars[8],
            'subjectID': loaded_vars[9],
            'StimPeriod': loaded_vars[10]
        }
    except Exception as e:
        print(f"Failed to load {session_id}: {e}")

print(f"\nCohort Loading Complete. Total animals loaded: {len(cohort_data)}")

#%% Average trials within an animal and merge
# 1. Aggregation: Create a summary dictionary for the cohort
cohort_summary = {}

for sid, data in cohort_data.items():
    rois = data['Roi2Vis']  # e.g., [0, 2, 5]
    psth_all = data['psth_data']
    peaks_all = data['peak_results']
    
    cohort_summary[sid] = {'G': {}, 'R': {}}
    
    # Process each trial type found in the PSTH data
    trial_types = [k.replace('G_', '').replace('_base', '') 
                   for k in psth_all.keys() if k.startswith('G_')]

    for tt in trial_types:
        for sig in ['G', 'R']:
            key = f"{sig}_{tt}_base"
            if key in psth_all:
                # psth_all[key] shape is (Time, ROIs, Trials)
                # 1. Subset the ROIs of interest
                roi_subset = psth_all[key][:, rois, :]
                
                # 2. Average across those ROIs to get (Time, Trials)
                subject_psth = np.nanmean(roi_subset, axis=1)
                
                # 3. Aggregate Peak Magnitudes for those ROIs
                peak_key = f"{key}_peak_mag"
                if peak_key in peaks_all:
                    # Average peak magnitude across ROIs for each trial
                    subject_peaks = np.nanmean(peaks_all[peak_key][rois, :], axis=0)
                else:
                    subject_peaks = np.array([])

                cohort_summary[sid][sig][tt] = {
                    'psth': subject_psth, 
                    'peaks': subject_peaks
                }
                
#%% plot avgd ROIs for each animal, compare animals to each other 
# 2. Plotting: Compare Green vs Red for a specific trial type across the cohort
target_tt = 'CS3R'  # Change this to any trial type you want to see
# Get time axis from the first animal in the cohort
first_sid = list(cohort_summary.keys())[0]
first_data = cohort_data[first_sid]
time_x = np.arange(cohort_summary[first_sid]['G'][target_tt]['psth'].shape[0]) / first_data['fs'] - first_data['preW'] / first_data['fs']

fig, axes = plt.subplots(len(cohort_summary), 1, figsize=(10, 4 * len(cohort_summary)), sharex=True)
if len(cohort_summary) == 1: axes = [axes]

for i, (sid, sig_data) in enumerate(cohort_summary.items()):
    ax = axes[i]
    
    for sig, color, label_prefix in zip(['G', 'R'], ['green', 'red'], ['Green', 'Red']):
        # psth shape is (Time, Trials)
        psth = sig_data[sig][target_tt]['psth']
        n_trials = psth.shape[1]
        
        # Calculate Mean and SEM
        mean_trace = np.nanmean(psth, axis=1)
        sem_trace = np.nanstd(psth, axis=1) / np.sqrt(n_trials)
        
        # Plot Mean Line
        ax.plot(time_x, mean_trace, color=color, lw=2, label=f'{label_prefix} (n={n_trials} trials)')
        
        # Plot SEM Shading
        ax.fill_between(time_x, mean_trace - sem_trace, mean_trace + sem_trace, 
                        color=color, alpha=0.2, edgecolor='none')
    
    ax.set_title(f"Subject: {sid} | Trial: {target_tt}", fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', lw=1, alpha=0.3)
    ax.set_ylabel("ΔF/F")
    ax.legend(loc='upper right', fontsize='small')

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

#%% plot combined animals (signal)

# 1. Define Trial Types to export
trial_types_to_plot = ['CS3R', 'CS3UR'] # Add any others you need

for target_tt in trial_types_to_plot:
    
    # --- FIGURE 1: SIGNAL TRACES (Grand Average PSTH) ---
    fig_trace, ax_t = plt.subplots(figsize=(8, 5))
    
    # Logic to gather traces (assumes cohort_summary is already populated)
    grand_psth_g = [np.nanmean(cohort_summary[sid]['G'][target_tt]['psth'], axis=1) for sid in cohort_summary]
    grand_psth_r = [np.nanmean(cohort_summary[sid]['R'][target_tt]['psth'], axis=1) for sid in cohort_summary]
    
    for data_arr, col, lbl in [(np.array(grand_psth_g), 'green', 'Green'), (np.array(grand_psth_r), 'red', 'Red')]:
        mu = np.nanmean(data_arr, axis=0)
        sem = np.nanstd(data_arr, axis=0) / np.sqrt(data_arr.shape[0])
        ax_t.plot(time_x, mu, color=col, lw=2, label=lbl)
        ax_t.fill_between(time_x, mu-sem, mu+sem, color=col, alpha=0.2, edgecolor='none')

    ax_t.set_title(f'Grand Average PSTH: {target_tt}')
    ax_t.set_ylabel('∆F/F')
    ax_t.set_xlabel('Peri-Event Time (s)')
    ax_t.legend(frameon=False)
    sns.despine(ax=ax_t)
    
    # Save Trace Figure
    trace_path = os.path.join(SaveDir, f'GrandAverage_PSTH_{target_tt}.svg')
    fig_trace.savefig(trace_path, format='svg', transparent=True, bbox_inches='tight')

    
    # AGGREGATE DATA FOR PEAK
    events = ['cs', 'rew', 'lick']
    event_labels = ['vs CS Onset', 'vs Reward', 'vs First Lick']
    
    mag_data = {'G': [], 'R': []}
    lat_data = {e: {'G': [], 'R': []} for e in events}

    for sid, data in cohort_data.items():
        rois = data['Roi2Vis']
        peaks_all = data['peak_results']
        
        for sig in ['G', 'R']:
            # Magnitude: One value per mouse (averaged over ROIs and Trials)
            m_key = f"{sig}_{target_tt}_base_peak_mag"
            if m_key in peaks_all:
                sub_mag = np.nanmean(np.nanmean(peaks_all[m_key][rois, :], axis=0))
                mag_data[sig].append(sub_mag)
            
            # Latencies: One value per mouse per event
            for event in events:
                l_key = f"{sig}_{target_tt}_base_peak_{event}_lat"
                if l_key in peaks_all:
                    sub_lat = np.nanmean(np.nanmean(peaks_all[l_key][rois, :], axis=0))
                    lat_data[event][sig].append(sub_lat)

    # --- FIGURE: PEAK MAGNITUDE & LATENCY ---
    fig_peaks = plt.figure(figsize=(12, 5))
    gs = fig_peaks.add_gridspec(1, 2, width_ratios=[1, 2.5])

    # Subplot A: Magnitude
    ax1 = fig_peaks.add_subplot(gs[0])
    for i, sig in enumerate(['G', 'R']):
        vals = mag_data[sig]
        if vals:
            mu, sem = np.nanmean(vals), np.nanstd(vals)/np.sqrt(len(vals))
            ax1.bar(i, mu, yerr=sem, color='green' if sig=='G' else 'red', alpha=0.5, capsize=5)
            ax1.scatter([i]*len(vals), vals, color='black', edgecolors='white', zorder=3, s=30)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(['Green', 'Red'])
    ax1.set_ylabel('Peak Amplitude (% ∆F/F)'); ax1.set_title('Peak Magnitudes')

    # Subplot B: Latencies
    ax2 = fig_peaks.add_subplot(gs[1])
    x_pos = np.arange(len(events))
    width = 0.3
    for i, event in enumerate(events):
        for j, sig in enumerate(['G', 'R']):
            vals = lat_data[event][sig]
            if vals:
                mu, sem = np.nanmean(vals), np.nanstd(vals)/np.sqrt(len(vals))
                p = i + (j - 0.5) * width
                ax2.bar(p, mu, width, yerr=sem, color='green' if sig=='G' else 'red', alpha=0.5, capsize=5)
                ax2.scatter([p]*len(vals), vals, color='black', edgecolors='white', zorder=3, s=20)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(event_labels)
    ax2.set_ylabel('Peak Latency (s)'); ax2.set_title('Event-Relative Peak Latencies')

    sns.despine()
    plt.suptitle(f"Cohort Peaks: {target_tt}", fontweight='bold')
    
    # Save the PDF
    peak_save_path = os.path.join(SaveDir, f'GrandAverage_Peaks_{target_tt}.svg')
    fig_peaks.savefig(peak_save_path, format='svg', transparent=True, bbox_inches='tight')
    



#%% Plot trial-by-trial peak comparison
print("\nRunning trial-by-trial correlation comparisons...")

for target_tt in trial_types_to_plot:
    all_trials_g_peaks = []
    all_trials_r_peaks = []

    for sid, data in cohort_data.items():
        # 1. Gather Amplitude Coupling Data (Scatter Plot)
        m_key_g = f"G_{target_tt}_base_peak_mag"
        m_key_r = f"R_{target_tt}_base_peak_mag"
        rois = data['Roi2Vis']
        
        if m_key_g in data['peak_results'] and m_key_r in data['peak_results']:
            # Trial-by-trial peaks (averaged across ROIs)
            g_matrix = data['peak_results'][m_key_g][rois, :]
            r_matrix = data['peak_results'][m_key_r][rois, :]
            
            # Flatten trials from all ROIs into a 1D array (3 ROIs x 20 trials becomes a 60-element vector)
            g_pts = g_matrix.flatten()
            r_pts = r_matrix.flatten()
            
            # save the peak magnitude for green and red for each trial
            all_trials_g_peaks.extend(g_pts)
            all_trials_r_peaks.extend(r_pts)



    # --- Plotting Panels ---
    fig, ax1 = plt.subplots(figsize=(5, 5))

    # Panel A: Amplitude Coupling
    x, y = np.array(all_trials_r_peaks), np.array(all_trials_g_peaks)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if len(x[mask]) > 1:
        slope, intercept, r_val, p_val, _ = stats.linregress(x[mask], y[mask])
        ax1.scatter(x[mask], y[mask], color='gray', alpha=0.3, s=15, edgecolors='none')
        ax1.plot(x[mask], slope*x[mask] + intercept, color='red', label=f'R²={r_val**2:.3f}')
        ax1.set_title(f'Amplitude Coupling ({target_tt})')
        ax1.set_xlabel('Red Peak (% ∆F/F)'); ax1.set_ylabel('Green Peak (% ∆F/F)')
        ax1.legend(frameon=False)


    sns.despine()
    plt.tight_layout()
    
    # Save results
    peakcomp_path = os.path.join(SaveDir, f'Peak-Amplitude_Trial-by-Trial_{target_tt}.svg')
    plt.savefig(peakcomp_path, format='svg', transparent=True)
    plt.show()




#%% plot cross-correlation
import random

print("\n--- Running Temporal Lag with Shuffled Control ---")
for target_tt in trial_types_to_plot:
    all_subject_xcorrs = []
    all_subject_shuffled = []
    fs = cohort_data[list(cohort_data.keys())[0]]['fs']

    for sid, data in cohort_data.items():
        g_psths = cohort_summary[sid]['G'][target_tt]['psth'] 
        r_psths = cohort_summary[sid]['R'][target_tt]['psth']
        
        n_trials = g_psths.shape[1]
        if n_trials < 2: continue # Need at least 2 trials to shuffle
        
        # 1. REAL CORRELATION (Trial i vs Trial i)
        trial_xcorrs = []
        for t in range(n_trials):
            g_n = (g_psths[:, t] - np.mean(g_psths[:, t])) / (np.std(g_psths[:, t]) + 1e-6)
            r_n = (r_psths[:, t] - np.mean(r_psths[:, t])) / (np.std(r_psths[:, t]) + 1e-6)
            trial_xcorrs.append(signal.correlate(g_n, r_n, mode='full') / len(g_n))
        
        # 2. SHUFFLED CONTROL (Trial i vs Trial j)
        shuffled_indices = list(range(n_trials))
        random.shuffle(shuffled_indices)
        
        shuff_xcorrs = []
        for i, j in enumerate(shuffled_indices):
            # Ensure we don't accidentally pick the same trial (i != j)
            if i == j: j = (j + 1) % n_trials 
            
            g_n = (g_psths[:, i] - np.mean(g_psths[:, i])) / (np.std(g_psths[:, i]) + 1e-6)
            r_n = (r_psths[:, j] - np.mean(r_psths[:, j])) / (np.std(r_psths[:, j]) + 1e-6)
            shuff_xcorrs.append(signal.correlate(g_n, r_n, mode='full') / len(g_n))
            
        all_subject_xcorrs.append(np.mean(trial_xcorrs, axis=0))
        all_subject_shuffled.append(np.mean(shuff_xcorrs, axis=0))

    # --- PLOTTING ---
    lag_times = np.arange(-(len(all_subject_xcorrs[0]) // 2), (len(all_subject_xcorrs[0]) // 2) + 1) / fs
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot Shuffled (Null)
    shuff_mu = np.nanmean(all_subject_shuffled, axis=0)
    ax.plot(lag_times, shuff_mu, color='gray', alpha=0.5, linestyle='--', label='Shuffled (Chance)')
    
    # Plot Real Data
    real_mu = np.nanmean(all_subject_xcorrs, axis=0)
    real_sem = np.nanstd(all_subject_xcorrs, axis=0) / np.sqrt(len(all_subject_xcorrs))
    ax.plot(lag_times, real_mu, color='purple', lw=2, label='Real Data')
    ax.fill_between(lag_times, real_mu-real_sem, real_mu+real_sem, color='purple', alpha=0.2)
    
    # Peak Annotation
    peak_lag = lag_times[np.argmax(real_mu)]
    ax.axvline(0, color='black', alpha=0.3)
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(peak_lag, color='red', linestyle=':', label=f'Peak Lag: {peak_lag*1000:.1f}ms')
    
    ax.set_title(f'Temporal Lag vs Shuffled Control: {target_tt}')
    ax.set_xlabel('Time (s)\n<--- Green Leads | Red Leads ---> ')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-0.1, 1.0])
    ax.legend(frameon=False)
    sns.despine()

    plt.savefig(os.path.join(SaveDir, f'CrossCorr_Shuffled_{target_tt}.svg'), format='svg')
    plt.show()
    
#%% Plot decay constant
# Storage for kinetics results
kinetics_results = []

for sid, data in cohort_data.items():
    for target_tt in trial_types_to_plot:
        # Get the mean PSTH for this subject/trial type
        # (Assuming these are the baseline-subtracted traces)
        g_mean = np.mean(cohort_summary[sid]['G'][target_tt]['psth'], axis=1)
        r_mean = np.mean(cohort_summary[sid]['R'][target_tt]['psth'], axis=1)
        
        # Define the post-event window to find the peak (e.g., first 3 seconds)
        search_window = (time_x >= 0) & (time_x <= 5)
        g_peak_idx = np.argmax(g_mean[search_window]) + np.where(search_window)[0][0]
        r_peak_idx = np.argmax(r_mean[search_window]) + np.where(search_window)[0][0]
        
        # Calculate Half-Lives
        g_t12 = fipf_p.calculate_half_life(time_x, g_mean, g_peak_idx)
        r_t12 = fipf_p.calculate_half_life(time_x, r_mean, r_peak_idx)
        
        kinetics_results.append({
            'Subject': sid,
            'TrialType': target_tt,
            'Green_HalfLife': g_t12,
            'Red_HalfLife': r_t12,
            'Difference': g_t12 - r_t12
        })

# Convert to DataFrame for easy viewing
import pandas as pd
df_kinetics = pd.DataFrame(kinetics_results)
print(df_kinetics.groupby('TrialType')[['Green_HalfLife', 'Red_HalfLife']].mean())

#%% plot moment to moment brightness SINGLE TRIAL
g_data = cohort_summary[sid]['G'][target_tt]['psth'][:, 1] # Trial 0
r_data = cohort_summary[sid]['R'][target_tt]['psth'][:, 1] # Trial 0
fipf_p.plot_brightness_scatter(g_data, r_data)

#%% plot moment to moment brightness ALL TRIALS
for sid, data in cohort_data.items():
    for target_tt in trial_types_to_plot:
        g_data = cohort_summary[sid]['G'][target_tt]['psth']
        r_data = cohort_summary[sid]['R'][target_tt]['psth'] 
        fipf_p.plot_global_moment_scatter(g_data, r_data, target_tt, sid)