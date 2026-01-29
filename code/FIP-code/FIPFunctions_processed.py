# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 15:42:19 2026

@author: carrie.stine
"""

import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#%% loading h5py data back in
def load_fip_h5(file_path):
    with h5py.File(file_path, 'r') as hf:
        # Attributes
        subjectID = hf.attrs['subjectID']
        sampling_rate = hf.attrs['sampling_rate']
        preW = hf.attrs['preW']
        StimPeriod = hf.attrs['StimPeriod']
        Roi2Vis = list(hf['Roi2Vis'][:])
        
        # --- Print Summary Statement ---
        print(f"\n{'='*40}")
        print(f"LOADING SESSION: {subjectID}")
        print(f"{'='*40}")
        print(f"File Source:  {os.path.basename(file_path)}")
        print(f"Parameters:   {sampling_rate} Hz | Pre-window: {preW} | Stim: {StimPeriod}s")
        print(f"ROIs to Evaluate:  {Roi2Vis}")

        # Load RT Data as single array
        rt_data = hf['rt_data'][:] if 'rt_data' in hf else None

        # Load Groups back into Dictionaries
        def h5_to_dict(group_name):
            d = {}
            if group_name in hf:
                for key in hf[group_name].keys():
                    d[key] = hf[group_name][key][:]
            return d

        psth_data = h5_to_dict('psth_data')
        psth_pooled_data = h5_to_dict('psth_pooled_data')
        peak_results = h5_to_dict('peak_results')
        TSdict = h5_to_dict('TSdict')
        TSdict_CSrewarded = h5_to_dict('TSdict_CSrewarded')
        
        # ---  Print Trial Breakdown ---
        print("-" * 40)
        print("Trial Counts Found in PSTH Data:")
        # We look at the Green channels as a representative for trial counts
        found_types = [k for k in psth_data.keys() if k.startswith('G_') and k.endswith('_base')]
        for kt in sorted(found_types):
            n_trials = psth_data[kt].shape[2]
            # Strip prefix/suffix for cleaner printing (e.g., G_CS1R_base -> CS1R)
            clean_name = kt.replace('G_', '').replace('_base', '')
            print(f"  {clean_name}: {n_trials} trials")
        print(f"{'='*40}\n")

    return (psth_data, psth_pooled_data, rt_data, peak_results, 
            TSdict, TSdict_CSrewarded, Roi2Vis, 
            sampling_rate, preW, subjectID, StimPeriod)



#%% plot subject-sorted heat maps
def plot_trial_by_trial_heatmaps(cohort_summary, trial_types, time_axis, save_dir):
    for tt in trial_types:
        g_all_trials = []
        r_all_trials = []
        dividers = []
        subject_label_positions = []
        current_row = 0

        # 1. Stack trials from all subjects
        for sid in cohort_summary.keys():
            # Get PSTH: shape is [Time x Trials]
            g_psth = cohort_summary[sid]['G'][tt]['psth']
            r_psth = cohort_summary[sid]['R'][tt]['psth']
            
            # Transpose to [Trials x Time] for the heatmap
            g_trials = g_psth.T
            r_trials = r_psth.T
            
            num_trials = g_trials.shape[0]
            
            g_all_trials.append(g_trials)
            r_all_trials.append(r_trials)
            
            # Keep track of where to draw the line between mice
            subject_label_positions.append(current_row + num_trials // 2)
            current_row += num_trials
            dividers.append(current_row)

        # Combine into giant matrices
        g_final = np.vstack(g_all_trials)
        r_final = np.vstack(r_all_trials)
        
        # Calculate global scale limits
        combined_data = np.concatenate([g_final, r_final])
        vmin_global = np.nanpercentile(combined_data, 1)
        vmax_global = np.nanpercentile(combined_data, 99)

        # 2. Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
        
        zero_idx = np.searchsorted(time_axis, 0)
        
        # Green Heatmap
        sns.heatmap(g_final, ax=ax1, cmap='viridis', 
                    vmin = vmin_global, vmax=vmax_global, 
                    robust=True, cbar_kws={'label': 'Green dFF'},
                    rasterized=True)
        ax1.set_title(f'Green - All Trials: {tt}')
        
        # Red Heatmap
        sns.heatmap(r_final, ax=ax2, cmap='magma', 
                    vmin = vmin_global, vmax=vmax_global, 
                    robust=True, cbar_kws={'label': 'Red dFF'},
                    rasterized=True)
        ax2.set_title(f'Red - All Trials: {tt}')

        # Formatting
        xticks = np.arange(0, len(time_axis), 20)
        xticklabels = np.round(time_axis[xticks], 1)
        
        for ax in [ax1, ax2]:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('Time from Event (s)')
            # Event line at t=0
            ax.axvline(x=zero_idx, color='white', linestyle='--', linewidth=2)
            
            # Add horizontal lines between subjects
            for d in dividers[:-1]:
                ax.axhline(y=d, color='white', linestyle='-', linewidth=1)

        # Label the subjects on the Y-axis
        ax1.set_yticks(subject_label_positions)
        ax1.set_yticklabels(list(cohort_summary.keys()), rotation=0)
        ax1.set_ylabel('Trials (Grouped by Subject)')

        plt.tight_layout()
    
        # Saves as high-res SVG for publications or PNG for quick viewing
        save_path = os.path.join(save_dir, f"SubjectSorted_Heatmap_{tt}.svg")
        plt.savefig(save_path, format='svg', transparent=True)
        print(f"Saved: {save_path}")
       
        plt.show()

#%% plot time interleaved heat maps
def plot_interleaved_chronological_heatmaps(cohort_summary, trial_types, time_axis, save_dir):
    for tt in trial_types:
        interleaved_g = []
        interleaved_r = []
        row_labels = []
        
        # 1. Determine the maximum number of trials any animal has
        sids = list(cohort_summary.keys())
        max_trials = max([cohort_summary[sid]['G'][tt]['psth'].shape[1] for sid in sids])

        # 2. Interleave: Loop through trial index first, then subjects
        for trial_idx in range(max_trials):
            for sid in sids:
                g_psth = cohort_summary[sid]['G'][tt]['psth']
                r_psth = cohort_summary[sid]['R'][tt]['psth']
                
                # Check if this animal actually has this trial index 
                # (in case one session was shorter than others)
                if trial_idx < g_psth.shape[1]:
                    # Extract trial [Time] and add as a row
                    interleaved_g.append(g_psth[:, trial_idx])
                    interleaved_r.append(r_psth[:, trial_idx])
                    row_labels.append(f"T{trial_idx+1}_{sid}")

        g_final = np.array(interleaved_g)
        r_final = np.array(interleaved_r)

        # 3. Calculate Global Scale (Synced)
        combined_data = np.concatenate([g_final, r_final])
        vmin_global = np.nanpercentile(combined_data, 1)
        vmax_global = np.nanpercentile(combined_data, 99)

        # 4. Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
        
        
        # Plotting
        sns.heatmap(g_final, ax=ax1, cmap='Greens', vmin=vmin_global, vmax=vmax_global,
                    cbar_kws={'label': '$\Delta F/F$'},
                    rasterized=True)
        ax1.set_title(f'Interleaved Green (DA) - {tt}')
        
        sns.heatmap(r_final, ax=ax2, cmap='Oranges', vmin=vmin_global, vmax=vmax_global,
                    cbar_kws={'label': '$\Delta F/F$'},
                    rasterized=True)
        ax2.set_title(f'Interleaved Red (Calcium) - {tt}')

        # Formatting
        xticks = np.arange(0, len(time_axis), 20)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(np.round(time_axis[xticks], 1))
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(np.round(time_axis[xticks], 1))


        plt.tight_layout()
        
        # Saves as high-res SVG for publications or PNG for quick viewing
        save_path = os.path.join(save_dir, f"Interleaved_Heatmap_{tt}.svg")
        plt.savefig(save_path, format='svg', transparent=True)
        print(f"Saved: {save_path}")
        
        plt.show()
        


#%% calculate decay constant
def calculate_half_life(time_axis, psth_trace, peak_idx):
    """Calculates the time it takes for a signal to decay by 50%."""
    if peak_idx >= len(psth_trace) - 1:
        return np.nan
    
    peak_val = psth_trace[peak_idx]
    half_max = peak_val / 2
    
    # Look at the trace after the peak
    decay_trace = psth_trace[peak_idx:]
    decay_time = time_axis[peak_idx:]
    
    # Find where the signal first drops below half-max
    below_half_max = np.where(decay_trace <= half_max)[0]
    
    if len(below_half_max) > 0:
        half_life_index = below_half_max[0]
        return decay_time[half_life_index] - decay_time[0]
    else:
        return np.nan # Signal didn't decay 50% within the window
    
#%% 2D correlation plot
def plot_brightness_scatter(g_trace, r_trace, title="Moment-to-Moment Coupling"):
    """
    g_trace: Z-scored or dFF Green signal
    r_trace: Z-scored or dFF Red signal
    """
    plt.figure(figsize=(6, 6))
    
    # Scatter plot with transparency to see density
    plt.scatter(r_trace, g_trace, alpha=0.1, color='purple', s=2)
    
    # Add a regression line to see the trend
    from scipy.stats import linregress
    mask = ~np.isnan(r_trace) & ~np.isnan(g_trace)
    slope, intercept, r_value, p_value, std_err = linregress(r_trace[mask], g_trace[mask])
    
    # Create points for the regression line
    x_range = np.array([np.min(r_trace[mask]), np.max(r_trace[mask])])
    line = slope * x_range + intercept
    
    plt.plot(x_range, line, color='red', linestyle='--', 
             label=f'Linear Fit (R²={r_value**2:.2f})')
    
    plt.xlabel('Red Signal')
    plt.ylabel('Green Signal')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.show()

#%%
def plot_global_moment_scatter(g_psths, r_psths, target_tt, sid):
    """
    g_psths: 2D array [Time x Trials] 
    r_psths: 2D array [Time x Trials]
    """
    # 1. Flatten all trials and all timepoints into two long vectors
    g_all = g_psths.flatten()
    r_all = r_psths.flatten()
    
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
    plt.title(f'Moment-to-Moment Coupling:{sid} {target_tt}\n(All Timepoints, All Trials)')
    plt.colorbar(label='Point Density')
    plt.legend()
    sns.despine()
    plt.show()
