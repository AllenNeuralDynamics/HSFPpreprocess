# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:25:17 2026

@author: carrie.stine
"""

import os
from pathlib import Path
import numpy as np
from natsort import natsorted
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

#%% constants
SAT_VAL = 7000 # saturation value of camera (12000 Smrithi, 7000 Carrie)
FIBER_WIDTH = 40 # width of fiber (in pixels) (60 Smrithi, 40 Carrie)
USE_LASER_1 = 0 # first laser used for calculating affine transformation (1 Smrithi, 0 Carrie)
USE_LASER_2 = 2 # second laser used for calculating affine transformation (2 Smrithi, 2 Carrie)
DIST_THRESH = 50 # distance between laser peaks (100 Smrithi, 50 Carrie)

#%% session ID
session_id = "HSFP_841357_20260416T115045"

#%% functions
def load_session_paths(data_dir, session_id):
    """Create paths to session data."""
    path = os.path.join(data_dir, session_id)
    calib_path = os.path.join(path, 'fib')
    return path, calib_path


    
    
    
    
    
    
    
    
    
    