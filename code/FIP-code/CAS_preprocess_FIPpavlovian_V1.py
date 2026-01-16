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
import matplotlib.gridspec as gridspec
import numpy as np
import csv
import glob
import re
from scipy.optimize import curve_fit
import json
import pandas as pd
from scipy.stats import sem

import PreprocessingFunctions2 as pf
import FIPFunctions2 as fipf


session_id = 'FIP_835527_2026-01-12_10-31-26'

SaveDir = r'C:\output_data\results\results_' + session_id
AnalDir = r'C:\output_data' + os.sep + session_id + os.sep + 'behavior'

os.makedirs(SaveDir, exist_ok=True)

# manually enter when the first reward trial that should actually be counted happened
first_rew_idx = 0           # 0 for 836733 12/3/25
                            # 23 for 836732 12/3/25
# choose which ROIs (fibers) to visualize
#Roi2Vis=[0,1,2]
Roi2Vis = [0,1]
AllPlot=0

# params for pre-processing
nFrame2cut = 100  #crop initial n frames
sampling_rate = 20 #individual channel (not total)
kernelSize = 1 #median filter
degree = 4 #polyfit
b_percentile = 0.70 #To calculare F0, median of bottom x%

StimPeriod = 0.5 #sec for visualization`
preW=100 #nframes for PSTH
LickWindow=5.0 #sec window length for Consummatory/Omission licks

#%% Load the data
data1, data2, data3, subjectID, TSdict = fipf.load_fip_data(AnalDir)


#%% Trim out initial manual rewards
TSdict['Reward'] = TSdict['Reward'][first_rew_idx:, :]


#%% Sync lengths and get session time
data1, data2, data3, PMts, time_seconds = fipf.sync_and_time(data1, data2, data3, sampling_rate)


#%% Preprocess the data
Ctrl_dF_F, G_dF_F, R_dF_F = fipf.preprocess_all_channels(
    data1, data2, data3, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile
)


#%% Extract event timestamps into frame indices
TSFramesdict = fipf.extract_trial_frames(TSdict, data1[:, 0])
RewardFrames = TSFramesdict.get('Reward', [])
LickFrames   = TSFramesdict.get('Lick', [])
CS1Frames   = TSFramesdict.get('CS1', [])
CS2Frames   = TSFramesdict.get('CS2', [])
CS3Frames   = TSFramesdict.get('CS3', [])


#%% Load pupil data (optional)
pupil_time, pupil_data = fipf.load_pupil_data(AnalDir, data1[0, 0])
#%% Plot the entire trace
events = {
    'Reward': RewardFrames,
    'CS1': CS1Frames,
    'CS2': CS2Frames,
    'CS3': CS3Frames,
    'Lick': LickFrames
}

fipf.plot_whole_trace(time_seconds, Ctrl_dF_F, G_dF_F, R_dF_F, Roi2Vis, events)

