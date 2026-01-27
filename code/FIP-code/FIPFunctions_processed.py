# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 15:42:19 2026

@author: carrie.stine
"""

import h5py
import numpy as np
import os
import glob
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