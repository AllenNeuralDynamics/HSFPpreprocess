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
from scipy import signal

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


#%% load cohort data
def load_cohort(session_ids, SessionDir):
    """Load FIP sessions from a list of session IDs and return cohort_data dict."""
    cohort_data = {}

    for session_id in session_ids:
        session_path = SessionDir + os.sep + session_id + os.sep + 'fib'
        print(f"Searching in: {session_path}")

        h5_files = glob.glob(os.path.join(session_path, "*.h5"))

        if not h5_files:
            if os.path.exists(session_path):
                print(f"Folder exists, but no .h5 found. Folder contains: {os.listdir(session_path)}")
            else:
                print(f"Error: The directory path does not exist: {session_path}")
            continue

        target_file = h5_files[0]
        print(f"Found file: {os.path.basename(target_file)}")

        try:
            loaded_vars = load_fip_h5(target_file)
            cohort_data[session_id] = {
                'psth_data':   loaded_vars[0],
                'psth_pooled': loaded_vars[1],
                'rt_data':     loaded_vars[2],
                'peak_results':loaded_vars[3],
                'TSdict':      loaded_vars[4],
                'TSdict_rew':  loaded_vars[5],
                'Roi2Vis':     loaded_vars[6],
                'fs':          loaded_vars[7],
                'preW':        loaded_vars[8],
                'subjectID':   loaded_vars[9],
                'StimPeriod':  loaded_vars[10]
            }
        except Exception as e:
            print(f"Failed to load {session_id}: {e}")

    print(f"\nCohort Loading Complete. Total animals loaded: {len(cohort_data)}")
    return cohort_data


#%% build the cohort summary
def build_cohort_summary(cohort_data):
    """Average trials within each animal and merge into a cohort summary dict."""
    cohort_summary = {}

    for sid, data in cohort_data.items():
        rois = data['Roi2Vis']
        psth_all = data['psth_data']
        peaks_all = data['peak_results']

        cohort_summary[sid] = {'G': {}, 'R': {}, 'C': {}}

        trial_types = [k.replace('G_', '').replace('_base', '')
                       for k in psth_all.keys() if k.startswith('G_')]

        for tt in trial_types:
            for sig in ['G', 'R', 'C']:
                key = f"{sig}_{tt}_base"
                if key in psth_all:
                    # psth_all[key] shape is (Time, ROIs, Trials)
                    roi_subset = psth_all[key][:, rois, :]
                    subject_psth = np.nanmean(roi_subset, axis=1)

                    peak_key = f"{key}_peak_mag"
                    if peak_key in peaks_all:
                        subject_peaks = np.nanmean(peaks_all[peak_key][rois, :], axis=0)
                    else:
                        subject_peaks = np.array([])

                    cohort_summary[sid][sig][tt] = {
                        'psth': subject_psth,
                        'peaks': subject_peaks
                    }

    return cohort_summary



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
        sns.heatmap(g_final, ax=ax1, cmap='Greens', 
                    # vmin=vmin_global, vmax=vmax_global,  # set shared colormap min/max
                    vmin=-5, vmax=20,
                    cbar_kws={'label': '$\Delta F/F$'},
                    rasterized=True)
        ax1.set_title(f'Interleaved Green (DA) - {tt}')
        
        sns.heatmap(r_final, ax=ax2, cmap='Oranges', 
                    # vmin=vmin_global, vmax=vmax_global,  # set shared colormap min/max
                    vmin=-5, vmax=20,
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
        

#%% plot trial by trial peak comparison
def plot_trial_peak_comparison(cohort_data, trial_types_to_plot, results_dir):
    """Plot trial-by-trial Green vs Red peak amplitude coupling for each trial type."""
    print("\nRunning trial-by-trial correlation comparisons...")

    for target_tt in trial_types_to_plot:
        all_trials_g_peaks = []
        all_trials_r_peaks = []

        for _, data in cohort_data.items():
            m_key_g = f"G_{target_tt}_base_peak_mag"
            m_key_r = f"R_{target_tt}_base_peak_mag"
            rois = data['Roi2Vis']

            if m_key_g in data['peak_results'] and m_key_r in data['peak_results']:
                g_pts = data['peak_results'][m_key_g][rois, :].flatten()
                r_pts = data['peak_results'][m_key_r][rois, :].flatten()
                all_trials_g_peaks.extend(g_pts)
                all_trials_r_peaks.extend(r_pts)

        _, ax1 = plt.subplots(figsize=(5, 5))

        x, y = np.array(all_trials_r_peaks), np.array(all_trials_g_peaks)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if len(x[mask]) > 1:
            slope, intercept, r_val, p_val, _ = stats.linregress(x[mask], y[mask])
            ax1.scatter(x[mask], y[mask], color='gray', alpha=0.3, s=15, edgecolors='none')
            ax1.plot(x[mask], slope * x[mask] + intercept, color='red',
                     label=f'R²={r_val**2:.3f}\n p={p_val}')
            ax1.set_title(f'Amplitude Coupling ({target_tt})')
            ax1.set_xlabel('Red Peak (% ∆F/F)')
            ax1.set_ylabel('Green Peak (% ∆F/F)')
            ax1.legend(frameon=False)

        sns.despine()
        plt.tight_layout()

        peakcomp_path = os.path.join(results_dir, f'Peak-Amplitude_Trial-by-Trial_{target_tt}.svg')
        plt.savefig(peakcomp_path, format='svg', transparent=True)
        plt.show()


#%% plot cross correlation
def plot_cross_correlation(cohort_summary, cohort_data, trial_types_to_plot, results_dir):
    """Compute and plot trial-averaged G/R cross-correlations. Returns source data dict keyed by trial type."""
    print("\n--- Running Temporal Lag Analysis ---")
    xcorr_data = {}

    for target_tt in trial_types_to_plot:
        all_subject_xcorrs = []
        all_subject_trial_xcorrs = {} # keyed by sid; each value is (Trials, Lags)
        all_subject_g_zscore = {}
        all_subject_r_zscore = {}
        # sampling rate is assumed to be consistent across sessions
        fs = cohort_data[list(cohort_data.keys())[0]]['fs']

        for sid in cohort_data:
            # Retrieve trial-averaged PSTHs for this subject; shape is (Time, Trials)
            g_psths = cohort_summary[sid]['G'][target_tt]['psth']
            r_psths = cohort_summary[sid]['R'][target_tt]['psth']

            n_trials = g_psths.shape[1]
            if n_trials < 2:
                continue
            
            # --- Compute trial-by-trial cross-correlations ---
            trial_xcorrs = []
            g_zscored_trials = []
            r_zscored_trials = []
            for t in range(n_trials):
                # Z-score each trial so correlation is amplitude-independent
                # A small epsilon (1e-6) prevents division by zero on flat traces
                g_n = (g_psths[:, t] - np.mean(g_psths[:, t])) / (np.std(g_psths[:, t]) + 1e-6)
                r_n = (r_psths[:, t] - np.mean(r_psths[:, t])) / (np.std(r_psths[:, t]) + 1e-6)
                g_zscored_trials.append(g_n)
                r_zscored_trials.append(r_n)
                # Full cross-correlation: output length is (2*N - 1); normalize
                # by N so the peak value equals the Pearson r at that lag
                trial_xcorrs.append(signal.correlate(g_n, r_n, mode='full') / len(g_n))

            # Preserve per-trial xcorrs for this subject before averaging; shape is (Trials, Lags)
            all_subject_trial_xcorrs[sid] = np.array(trial_xcorrs)
            # Preserve Z-scored signals that fed into the xcorr; shape (Trials, Time)
            all_subject_g_zscore[sid] = np.array(g_zscored_trials)
            all_subject_r_zscore[sid] = np.array(r_zscored_trials)
            
            # Average cross-correlations across trials to get one curve per subject
            all_subject_xcorrs.append(np.mean(trial_xcorrs, axis=0))

        # Build the lag axis in seconds: 0 = no lag, negative means green leads, positive means red leads
        lag_times = np.arange(-(len(all_subject_xcorrs[0]) // 2),
                               (len(all_subject_xcorrs[0]) // 2) + 1) / fs
        
        # Grand average and SEM across subjects
        real_mu  = np.nanmean(all_subject_xcorrs, axis=0)
        real_sem = np.nanstd(all_subject_xcorrs, axis=0) / np.sqrt(len(all_subject_xcorrs))
        
        # Lag (in seconds) at which the cross-correlation peaks
        peak_lag = lag_times[np.argmax(real_mu)]

        # --- Plot ---
        _, ax = plt.subplots(figsize=(7, 5))
        ax.plot(lag_times, real_mu, color='purple', lw=2, label='Real Data')
        ax.fill_between(lag_times, real_mu - real_sem, real_mu + real_sem, color='purple', alpha=0.2)
        ax.axvline(0, color='black', alpha=0.3)
        ax.axhline(0, color='black', alpha=0.3)
        ax.axvline(peak_lag, color='red', linestyle=':', label=f'Peak Lag: {peak_lag*1000:.1f}ms')
        ax.set_title(f'Temporal Lag: {target_tt}')
        ax.set_xlabel('Time (s)\n<--- Green Leads | Red Leads --->')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-0.1, 1.0])
        ax.legend(frameon=False)
        sns.despine()

        plt.savefig(os.path.join(results_dir, f'CrossCorr_{target_tt}.svg'), format='svg')
        plt.show()

        # Collect source data for downstream use/export
        xcorr_data[target_tt] = {
            'lag_times':      lag_times,
            'g_zscore':       all_subject_g_zscore,      # per-subject, per-trial Z-scored G; dict of (Trials, Time)
            'r_zscore':       all_subject_r_zscore,      # per-subject, per-trial Z-scored R; dict of (Trials, Time)
            'trial_xcorrs':   all_subject_trial_xcorrs,     # per-subject, per-trial
            'subject_xcorrs': all_subject_xcorrs,           # per-subject, trial-averaged
            'real_mu':        real_mu,
            'real_sem':       real_sem,
            'peak_lag':       peak_lag,
        }

    return xcorr_data

#%% plot cross correlation for individual trials (both ROIs)
def plot_cross_correlation_all_trials(cohort_summary, cohort_data, trial_types_to_plot, results_dir):
    """Compute and G/R cross-correlations for all trials. Returns source data dict keyed by trial type."""
    print("\n--- Running Temporal Lag Analysis ---")
    xcorr_data = {}

    for target_tt in trial_types_to_plot:
        all_subject_xcorrs = []
        all_subject_trial_xcorrs = {}  # keyed by sid; each value is (Trials, Lags)
        all_subject_g_zscore = {}      # keyed by sid; each value is (Trials, Time)
        all_subject_r_zscore = {}      # keyed by sid; each value is (Trials, Time)
        # Sampling rate is assumed to be consistent across sessions
        fs = cohort_data[list(cohort_data.keys())[0]]['fs']

        g_lookup = 'G_' + target_tt + '_base'
        r_lookup = 'R_' + target_tt + '_base'

        for sid in cohort_data:
            # Retrieve trial-averaged PSTHs for this subject; shape is (Time, Trials)
            g_psths = cohort_data[sid]['psth_pooled'][g_lookup]
            r_psths = cohort_data[sid]['psth_pooled'][r_lookup]

            # remove the dimensions of size 1
            g_psths = g_psths.squeeze()
            r_psths = r_psths.squeeze()

            n_trials = g_psths.shape[1]
            if n_trials < 2:
                continue

            # --- Compute trial-by-trial cross-correlations ---
            trial_xcorrs = []
            g_zscored_trials = []
            r_zscored_trials = []
            for t in range(n_trials):
                # Z-score each trial so correlation is amplitude-independent
                # A small epsilon (1e-6) prevents division by zero on flat traces
                g_n = (g_psths[:, t] - np.mean(g_psths[:, t])) / (np.std(g_psths[:, t]) + 1e-6)
                r_n = (r_psths[:, t] - np.mean(r_psths[:, t])) / (np.std(r_psths[:, t]) + 1e-6)
                g_zscored_trials.append(g_n)
                r_zscored_trials.append(r_n)
                # Full cross-correlation: output length is (2*N - 1); normalise by N
                # so the peak value equals the Pearson r at that lag
                trial_xcorrs.append(signal.correlate(g_n, r_n, mode='full') / len(g_n))

            # Preserve per-trial xcorrs for this subject before averaging; shape (Trials, Lags)
            all_subject_trial_xcorrs[sid] = np.array(trial_xcorrs)
            # Preserve Z-scored signals that fed into the xcorr; shape (Trials, Time)
            all_subject_g_zscore[sid] = np.array(g_zscored_trials)
            all_subject_r_zscore[sid] = np.array(r_zscored_trials)

            # Average cross-correlations across trials to get one curve per subject
            all_subject_xcorrs.append(np.mean(trial_xcorrs, axis=0))

        # Build the lag axis in seconds: 0 = no lag, negative = Green leads, positive = Red leads
        lag_times = np.arange(-(len(all_subject_xcorrs[0]) // 2),
                               (len(all_subject_xcorrs[0]) // 2) + 1) / fs

        # Grand average and SEM across subjects
        real_mu  = np.nanmean(all_subject_xcorrs, axis=0)
        real_sem = np.nanstd(all_subject_xcorrs, axis=0) / np.sqrt(len(all_subject_xcorrs))

        # Lag (in seconds) at which the cross-correlation peaks
        peak_lag = lag_times[np.argmax(real_mu)]

        # --- Plot ---
        _, ax = plt.subplots(figsize=(7, 5))
        ax.plot(lag_times, real_mu, color='purple', lw=2, label='Real Data')
        # Shaded band shows ± 1 SEM across subjects
        ax.fill_between(lag_times, real_mu - real_sem, real_mu + real_sem, color='purple', alpha=0.2)
        ax.axvline(0, color='black', alpha=0.3)       # reference: zero lag
        ax.axhline(0, color='black', alpha=0.3)       # reference: zero correlation
        ax.axvline(peak_lag, color='red', linestyle=':', label=f'Peak Lag: {peak_lag*1000:.1f}ms')
        ax.set_title(f'Temporal Lag: {target_tt}')
        ax.set_xlabel('Time (s)\n<--- Green Leads | Red Leads --->')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-0.1, 1.0])
        ax.legend(frameon=False)
        sns.despine()

        plt.savefig(os.path.join(results_dir, f'CrossCorr_{target_tt}.svg'), format='svg')
        plt.show()

        # Collect source data for downstream use or export
        xcorr_data[target_tt] = {
            'lag_times':      lag_times,
            'g_zscore':       all_subject_g_zscore,      # per-subject, per-trial Z-scored G; dict of (Trials, Time)
            'r_zscore':       all_subject_r_zscore,      # per-subject, per-trial Z-scored R; dict of (Trials, Time)
            'trial_xcorrs':   all_subject_trial_xcorrs,  # per-subject, per-trial xcorr; dict of (Trials, Lags)
            'subject_xcorrs': all_subject_xcorrs,        # per-subject average curves
            'real_mu':        real_mu,
            'real_sem':       real_sem,
            'peak_lag':       peak_lag,
        }

    return xcorr_data



#%% plot cross-correlation FINAL version used in paper 
def plot_cross_correlation_final(cohort_data, trial_types_to_plot, results_dir):
    """
    Compute G/R cross-correlations from raw per-ROI psth_data.

    Pipeline per trial type:
      1. Extract (Time, ROI, Trial) arrays directly from cohort_data['psth_data']
      2. Restrict to Roi2Vis (valid recorded ROIs) for each subject
      3. Z-score each trial for each ROI
      4. Cross-correlate G vs R per trial per ROI
      5. Average across trials  -> one curve per ROI
      6. Average across ROIs    -> one curve per subject
      7. Average across subjects -> grand mean +/- SEM for plotting

    Returns source data dict keyed by trial type.
    """
    print("\n--- Running Temporal Lag Analysis (per-ROI) ---")
    xcorr_data = {}

    for target_tt in trial_types_to_plot:
        g_key = f'G_{target_tt}_base'
        r_key = f'R_{target_tt}_base'

        fs = cohort_data[list(cohort_data.keys())[0]]['fs']

        # Nested storage: sid -> roi -> array
        g_zscore_all     = {}   # (Trials, Time)  per ROI per subject
        r_zscore_all     = {}   # (Trials, Time)  per ROI per subject
        trial_xcorrs_all = {}   # (Trials, Lags)  per ROI per subject
        roi_xcorrs_all   = {}   # (Lags,)         trial-averaged, per ROI per subject
        subject_xcorrs   = []   # (Lags,)         ROI-averaged, one per subject
        subject_sids     = []   # sid order matching subject_xcorrs

        for sid, data in cohort_data.items():
            rois      = data['Roi2Vis']   # indices of valid ROIs for this subject
            psth_data = data['psth_data']

            if g_key not in psth_data or r_key not in psth_data:
                print(f"  Skipping {sid}: missing {g_key} or {r_key}")
                continue

            # Shape: (Time, TotalROIs, Trials)
            g_all    = psth_data[g_key]
            r_all    = psth_data[r_key]
            n_trials = g_all.shape[2]

            if n_trials < 2:
                continue

            g_zscore_all[sid]     = {}
            r_zscore_all[sid]     = {}
            trial_xcorrs_all[sid] = {}
            roi_xcorrs_all[sid]   = {}
            roi_mean_xcorrs       = []  # collects one trial-averaged curve per ROI

            for roi in rois:
                g_trials_zscored = []
                r_trials_zscored = []
                this_roi_xcorrs  = []

                for t in range(n_trials):
                    g_t = g_all[:, roi, t]
                    r_t = r_all[:, roi, t]

                    # Z-score so correlation is amplitude-independent;
                    # epsilon prevents divide-by-zero on flat traces
                    g_n = (g_t - np.mean(g_t)) / (np.std(g_t) + 1e-6)
                    r_n = (r_t - np.mean(r_t)) / (np.std(r_t) + 1e-6)

                    g_trials_zscored.append(g_n)
                    r_trials_zscored.append(r_n)

                    # Normalise by N so peak approximates Pearson r at that lag
                    this_roi_xcorrs.append(signal.correlate(g_n, r_n, mode='full') / len(g_n))

                # Store per-trial data for this ROI; rows = trials
                g_zscore_all[sid][roi]     = np.array(g_trials_zscored)  # (Trials, Time)
                r_zscore_all[sid][roi]     = np.array(r_trials_zscored)  # (Trials, Time)
                trial_xcorrs_all[sid][roi] = np.array(this_roi_xcorrs)   # (Trials, Lags)

                # Step 5: average across trials -> one curve for this ROI
                roi_mean = np.mean(this_roi_xcorrs, axis=0)
                roi_xcorrs_all[sid][roi] = roi_mean
                roi_mean_xcorrs.append(roi_mean)

            # Step 6: average across valid ROIs -> one curve for this subject
            subject_xcorrs.append(np.mean(roi_mean_xcorrs, axis=0))
            subject_sids.append(sid)

        # Step 7: grand average and SEM across subjects
        lag_times = np.arange(-(len(subject_xcorrs[0]) // 2),
                               (len(subject_xcorrs[0]) // 2) + 1) / fs
        real_mu  = np.nanmean(subject_xcorrs, axis=0)
        real_sem = np.nanstd(subject_xcorrs, axis=0) / np.sqrt(len(subject_xcorrs))
        peak_lag = lag_times[np.argmax(real_mu)]

        # --- Plot ---
        _, ax = plt.subplots(figsize=(5, 5))
        ax.plot(lag_times, real_mu, color='purple', lw=2, label='Cross-Correlation')
        ax.fill_between(lag_times, real_mu - real_sem, real_mu + real_sem, color='purple', alpha=0.2)
        ax.axvline(0, color='black', alpha=0.3)       # reference: zero lag
        ax.axhline(0, color='black', alpha=0.3)       # reference: zero correlation
        ax.axvline(peak_lag, color='red', linestyle=':', label=f'Peak Lag: {peak_lag*1000:.1f}ms')
        ax.set_title(f'Temporal Lag: {target_tt}')
        ax.set_xlabel('Time (s)\n<--- Green Leads | Red Leads --->')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-0.1, 1.0])
        ax.legend(frameon=False)
        sns.despine()

        plt.savefig(os.path.join(results_dir, f'CrossCorr_Final_{target_tt}.svg'), format='svg')
        plt.show()

        # --- Per-subject subplot (1 x N) ---
        n_sids = len(subject_sids)
        _, axes = plt.subplots(1, n_sids, figsize=(5 * n_sids, 5), sharey=True)
        if n_sids == 1:
            axes = [axes]  # ensure iterable when only one subject

        for ax_s, sid, xcorr in zip(axes, subject_sids, subject_xcorrs):
            # SEM across ROIs for this subject
            roi_curves = np.array(list(roi_xcorrs_all[sid].values()))  # (n_rois, Lags)
            n_rois = roi_curves.shape[0]
            sid_sem = np.nanstd(roi_curves, axis=0) / np.sqrt(n_rois) if n_rois > 1 else np.zeros_like(xcorr)

            sid_peak_lag = lag_times[np.argmax(xcorr)]
            ax_s.plot(lag_times, xcorr, color='purple', lw=2)
            # ax_s.fill_between(lag_times, xcorr - sid_sem, xcorr + sid_sem, color='purple', alpha=0.2, edgecolor='none')
            ax_s.axvline(0, color='black', alpha=0.3)
            ax_s.axhline(0, color='black', alpha=0.3)
            ax_s.axvline(sid_peak_lag, color='red', linestyle=':', label=f'Peak: {sid_peak_lag*1000:.1f}ms')
            ax_s.set_title(cohort_data[sid]['subjectID'])
            ax_s.set_xlabel('Time (s)\n<--- Green Leads | Red Leads --->')
            ax_s.set_xlim([-1.0, 1.0])
            ax_s.set_ylim([-0.1, 1.0])
            ax_s.legend(frameon=False, fontsize=8)
            sns.despine(ax=ax_s)

        axes[0].set_ylabel('Correlation Coefficient')
        plt.suptitle(f'Per-Subject Temporal Lag: {target_tt}', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'CrossCorr_Final_PerSubject_{target_tt}.svg'), format='svg')
        plt.show()

        xcorr_data[target_tt] = {
            'lag_times':      lag_times,
            'g_zscore':       g_zscore_all,       # sid -> {roi -> (Trials, Time)}
            'r_zscore':       r_zscore_all,       # sid -> {roi -> (Trials, Time)}
            'trial_xcorrs':   trial_xcorrs_all,   # sid -> {roi -> (Trials, Lags)}
            'roi_xcorrs':     roi_xcorrs_all,     # sid -> {roi -> (Lags,)}
            'subject_xcorrs': subject_xcorrs,     # list of (Lags,), order matches subject_sids
            'subject_sids':   subject_sids,
            'real_mu':        real_mu,
            'real_sem':       real_sem,
            'peak_lag':       peak_lag,
        }

    return xcorr_data


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
