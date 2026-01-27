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
#%% plot combined ROIs for each animal, compare animals to each other 
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
    ax_t.legend(frameon=False)
    sns.despine(ax=ax_t)
    
    # Save Trace Figure
    trace_path = os.path.join(SaveDir, f'GrandAverage_PSTH_{target_tt}.pdf')
    fig_trace.savefig(trace_path, transparent=True, bbox_inches='tight')

    
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
    ax1.set_ylabel('Peak Mag (Z-score)'); ax1.set_title('Magnitudes')

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
    ax2.set_ylabel('Latency (s)'); ax2.set_title('Latencies')

    sns.despine()
    plt.suptitle(f"Cohort Peaks: {target_tt}", fontweight='bold')
    
    # Save the PDF
    peak_save_path = os.path.join(SaveDir, f'GrandAverage_Peaks_{target_tt}.pdf')
    fig_peaks.savefig(peak_save_path, transparent=True, bbox_inches='tight')
    










#%%
target_tt = 'CS3R'  # The trial type to average across the cohort

# 1. Collect one mean trace per subject
grand_psth_g = []
grand_psth_r = []

for sid in cohort_summary:
    # Get the mean trace for this animal (averaging its trials)
    subject_mean_g = np.nanmean(cohort_summary[sid]['G'][target_tt]['psth'], axis=1)
    subject_mean_r = np.nanmean(cohort_summary[sid]['R'][target_tt]['psth'], axis=1)
    
    grand_psth_g.append(subject_mean_g)
    grand_psth_r.append(subject_mean_r)

# Convert lists to 2D arrays: shape (Subjects, Time)
grand_psth_g = np.array(grand_psth_g)
grand_psth_r = np.array(grand_psth_r)

n_subjects = grand_psth_g.shape[0]

# 2. Plotting
plt.figure(figsize=(10, 6))

for data_arr, color, label in [(grand_psth_g, 'green', 'Green Signal'), 
                               (grand_psth_r, 'red', 'Red Signal')]:
    
    # Calculate Grand Mean and Inter-Subject SEM
    mean_trace = np.nanmean(data_arr, axis=0)
    sem_trace = np.nanstd(data_arr, axis=0) / np.sqrt(n_subjects)
    
    # Plotting
    plt.plot(time_x, mean_trace, color=color, lw=3, label=f'{label} (N={n_subjects} mice)')
    plt.fill_between(time_x, mean_trace - sem_trace, mean_trace + sem_trace, 
                     color=color, alpha=0.2, edgecolor='none')

# Formatting
plt.title(f"Grand Average: {target_tt}", fontsize=14, fontweight='bold')
plt.axvline(0, color='black', linestyle='--', alpha=0.6)
plt.axhline(0, color='black', lw=1, alpha=0.3)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Normalized Magnitude", fontsize=12)
plt.legend(frameon=False)
sns.despine() # Makes the plot look cleaner/published
plt.show()

#%% plot peak results for combined animals
target_tt = 'CS3R'
events = ['cs', 'rew', 'lick']
event_labels = ['vs CS Onset', 'vs Reward', 'vs First Lick']

# Storage
# mag_data will only have 'G' and 'R' (since magnitude is independent of event alignment)
mag_data = {'G': [], 'R': []}
# lat_data will be nested by event
lat_data = {e: {'G': [], 'R': []} for e in events}

for sid, data in cohort_data.items():
    rois = data['Roi2Vis']
    peaks_all = data['peak_results']
    
    for sig in ['G', 'R']:
        # 1. Handle Magnitudes (One per signal type)
        m_key = f"{sig}_{target_tt}_base_peak_mag"
        if m_key in peaks_all:
            # Average ROIs -> Average Trials -> One value per mouse
            sub_mag = np.nanmean(np.nanmean(peaks_all[m_key][rois, :], axis=0))
            mag_data[sig].append(sub_mag)
        
        # 2. Handle Latencies (Relative to each event)
        for event in events:
            l_key = f"{sig}_{target_tt}_base_peak_{event}_lat"
            if l_key in peaks_all:
                # Average ROIs -> Average Trials -> One value per mouse
                sub_lat = np.nanmean(np.nanmean(peaks_all[l_key][rois, :], axis=0))
                lat_data[event][sig].append(sub_lat)

# --- Plotting ---
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.5])

# Plot A: Peak Magnitude (Only 2 bars: Green vs Red)
ax1 = fig.add_subplot(gs[0])
m_colors = ['green', 'red']
for i, sig in enumerate(['G', 'R']):
    vals = mag_data[sig]
    mu, sem = np.nanmean(vals), np.nanstd(vals)/np.sqrt(len(vals))
    ax1.bar(i, mu, yerr=sem, color=m_colors[i], alpha=0.6, capsize=5)
    ax1.scatter([i]*len(vals), vals, color='black', edgecolors='white', zorder=3)

ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Green', 'Red'])
ax1.set_ylabel('Peak Magnitude (Z-score)')
ax1.set_title('Subject Magnitudes', fontweight='bold')

# Plot B: Latencies (3 pairs of bars)
ax2 = fig.add_subplot(gs[1])
x = np.arange(len(events))
width = 0.3

for i, event in enumerate(events):
    for j, (sig, col) in enumerate(zip(['G', 'R'], ['green', 'red'])):
        vals = lat_data[event][sig]
        if vals:
            mu, sem = np.nanmean(vals), np.nanstd(vals)/np.sqrt(len(vals))
            pos = i + (j - 0.5) * width
            ax2.bar(pos, mu, width, yerr=sem, color=col, alpha=0.6, capsize=5)
            ax2.scatter([pos]*len(vals), vals, color='black', edgecolors='white', zorder=3, s=20)

ax2.set_xticks(x)
ax2.set_xticklabels(event_labels)
ax2.set_ylabel('Latency to Peak (s)')
ax2.set_title('Latency Relative to Events', fontweight='bold')
ax2.legend(['Green', 'Red'], loc='upper right')

sns.despine()
plt.tight_layout()
plt.show()