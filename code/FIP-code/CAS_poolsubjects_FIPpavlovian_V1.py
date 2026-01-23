# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:55:54 2026

@author: carrie.stine
"""
import h5py
import numpy as np
import os


#%% EXAMPLES loading h5py data back in
def load_psth_h5(file_path):
    psth_reloaded = {}
    with h5py.File(file_path, 'r') as hf:
        # Get metadata
        sub_id = hf.attrs.get('subjectID')
        fs = hf.attrs.get('sampling_rate')
        
        # Load all datasets back into a dictionary
        for key in hf.keys():
            psth_reloaded[key] = hf[key][:] # [:] loads the data into a numpy array
            
    return psth_reloaded, sub_id, fs

# Usage
# data_dict, s_id, s_rate = load_psth_h5('your_file.h5')


with h5py.File(h5_filename, 'r') as hf:
    # Get all peak times for ROI 0 across all 25 trials for CS3R
    roi0_peaks = hf['peak_latencies/G_CS3R_base'][0, :]
    
    print(f"Mean peak time for ROI 0: {np.mean(roi0_peaks):.2f}s")
    print(f"Number of trials: {len(roi0_peaks)}")