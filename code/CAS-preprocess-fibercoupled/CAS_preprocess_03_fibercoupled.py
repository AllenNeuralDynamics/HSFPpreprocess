# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 09:48:49 2025

@author: carrie.stine
"""

#%% setup/imports
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple
from natsort import natsorted
import ast
import matplotlib.pyplot as plt

#%% variables/constants
# store the session id
session_id = 'HSFP_775510_2025-02-20_11-08-27'

#%% functions
def _correct_16bit_rollover(framestamps: np.ndarray, modulo: int = 2**16) -> tuple[np.ndarray, int]:
    """
    Correct 16-bit rollover in a sequence of framestamps.
    Assumes framestamps are modulo `modulo` (default 65536).
    Returns a monotonically non-decreasing int64 array starting at framestamps[0].
    """
    fs = np.asarray(framestamps)
    if fs.size == 0:
        return fs.astype(np.int64), 0
    
    # Ensure integer type
    fs = fs.astype(np.int64)

    # Compute stepwise differences; when diff is negative, add modulo to that step
    diffs = np.diff(fs, prepend=fs[0]).astype(np.int64)
    
    # Detect rollovers (negative diffs)
    negatives = diffs < 0
    rollovers = int(np.count_nonzero(negatives))
    
    # Handle rollovers by adding modulo
    diffs[negatives] += modulo

    corrected = np.cumsum(diffs)
    return corrected, rollovers



def load_session_params(
    fib_dir: Path
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[Path]]:
    """
    Load all session_params_[Timestamp].csv files in `fib_dir` and return:
      - numFrames: np.ndarray of length n (frames per CSV)
      - time: list of length n; each entry is np.ndarray of camera timestamps (s + us/1e6)
      - Frames: list of length n; each entry is np.ndarray of rollover-corrected framestamps
      - csv_paths: the ordered list of CSV Path objects processed (for traceability)

    Notes:
      - Expects columns: 'Framestamp', 'CameraTimestampSeconds', 'CameraTimestampMicroseconds'.
      - Will tolerate minor variations in case/underscores for these names.
      - If your microseconds column is named differently, add it to the candidates list below.
    """
    fib_dir = Path(fib_dir)
    #csv_paths = sorted(
    #    fib_dir.glob("session_params_*.csv"),
    #    key=_extract_timestamp_key
    # )
    csv_paths = natsorted(list(fib_dir.glob("session*.csv")))

    if not csv_paths:
        raise FileNotFoundError(
            f"No session_params_*.csv files found in {fib_dir}"
        )

    numFrames_list = []
    time_list: List[np.ndarray] = []
    frames_list: List[np.ndarray] = []

    for csv_path in csv_paths:
        # Read; dtype left flexible because headers can vary
        df = pd.read_csv(csv_path)

        # Resolve required columns 
        framestamp_col = "Framestamp"
        cam_sec_col = "CameraTimestampSeconds"
        cam_us_col = "CameraTimestampMicroSeconds"

        # Number of frames = number of rows
        n_rows = len(df)

        # Build time: seconds + microseconds / 1e6 (float64)
        # Ensure numeric types; coerce errors to NaN if present
        cols = df.columns
        if {"CameraTimestampSeconds", "CameraTimestampMicroSeconds"}.issubset(cols):
            cam_sec = pd.to_numeric(df[cam_sec_col], errors="coerce").to_numpy(dtype=np.float64)
            cam_us = pd.to_numeric(df[cam_us_col], errors="coerce").to_numpy(dtype=np.float64)
            time_arr = cam_sec + (cam_us / 1e6)
        elif "CameraTimestamp" in cols:
            time_arr =  pd.to_numeric(df["CameraTimestamp"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            raise ValueError("Missing CameraTimestamp metadata")
        
        # Framestamps: correct 16-bit rollover
        framestamps = pd.to_numeric(df[framestamp_col], errors="coerce").to_numpy(dtype=np.int64)
        frames_corrected, rollovers = _correct_16bit_rollover(framestamps, modulo=2**16)
        
        # Print rollover info
        print(f"{csv_path.name}: {rollovers} rollovers corrected")
        
        # EXCLUDE FINAL ENTRY (last framestamp is often corrupt/partially collected)
        # if there are at least two framestamps, exclude the last entry 
        if n_rows >= 1:
            numFrames_list.append(n_rows - 1)
            time_list.append(time_arr[:-1])
            frames_list.append(frames_corrected[:-1])
        # if there is only one framestamp in the file, do not exclude 'final' entry
        else:
            numFrames_list.append(n_rows)
            time_list.append(time_arr)
            frames_list.append(frames_corrected)

    # convert numFrames to a numpy array
    numFrames = np.array(numFrames_list, dtype=np.int64)
    return numFrames, time_list, frames_list, csv_paths



def check_dropped_frames(frames,times):
    """
    frames: np.ndarray of corrected framestamps
    times: np.ndarray of camera timestamps
    """
    # Calculate the jump between consecutive frames
    # diff[i] = frames[i+1] - frames[i]
    jumps = np.diff(frames)
    
    # Find where the jump is not 1
    drop_indices = np.where(jumps > 1)[0]
    
    if len(drop_indices) == 0:
        print("No dropped frames detected.")
        return
    
    for idx in drop_indices:
        drop_count = jumps[idx] - 1
        drop_time = times[idx]
        print(f"DROPPED {drop_count} frame(s) at time {drop_time:.3f}s (Index: {idx})")
        return drop_count, drop_time


def plot_timing_diagnostics(times):
    ifi = np.diff(times) * 1000  # Convert to milliseconds
    
    plt.figure(figsize=(10, 4))
    plt.plot(ifi, label='Inter-Frame Interval')
    plt.axhline(y=np.median(ifi), color='r', linestyle='--', label='Median IFI')
    plt.xlabel('Frame Index')
    plt.ylabel('Time Delta (ms)')
    plt.title('Timing Stability (Jitter Analysis)')
    plt.legend()
    plt.show()
    
    

def load_calib_values(calib_file, Xoffset, Yoffset):
    # Safe parsing of calibration.txt
    calib_dict = {}
    with open(calib_file, 'r') as f:
        for line in f:
            name, value = line.strip().split(' = ')
            calib_dict[name] = ast.literal_eval(value)
    
    # store values as local variables
    theta_r = calib_dict['rot_tform_thetaR'] 
    aff_tform_pt1 = calib_dict['aff_tform_pt1']
    aff_tform_pt2 = calib_dict['aff_tform_pt2']
    aff_tform_pt3 = calib_dict['aff_tform_pt3']
    aff_tform_pt4 = calib_dict['aff_tform_pt4'] 
    aff_tform_pt5 = calib_dict['aff_tform_pt5']
    aff_tform_pt6 = calib_dict['aff_tform_pt6']
    fiber1_pixels = calib_dict['fiber1_pixels']
    fiber2_pixels = calib_dict['fiber2_pixels']
    # calib_Xoffset = calib_dict['calib_Xoffset']
    
    # convert transformation points and fiber locations to pixels using X and Y offsets
    pt1 = [aff_tform_pt1[0]-Xoffset, aff_tform_pt1[1]-Yoffset] 
    pt2 = [aff_tform_pt2[0]-Xoffset, aff_tform_pt2[1]-Yoffset] 
    pt3 = [aff_tform_pt3[0]-Xoffset, aff_tform_pt3[1]-Yoffset] 
    pt4 = [aff_tform_pt4[0]-Xoffset, aff_tform_pt4[1]-Yoffset] 
    pt5 = [aff_tform_pt5[0]-Xoffset, aff_tform_pt5[1]-Yoffset] 
    pt6 = [aff_tform_pt6[0]-Xoffset, aff_tform_pt6[1]-Yoffset]
    fiber1_location = [fiber1_pixels[1]-Yoffset,fiber1_pixels[0]-Yoffset]
    fiber2_location = [fiber2_pixels[1]-Yoffset,fiber2_pixels[0]-Yoffset]

    fiber1_location = [int(x) for x in fiber1_location]
    fiber2_location = [int(x) for x in fiber2_location]
    
    return theta_r, pt1, pt2, pt3, pt4, pt5, pt6, fiber1_location, fiber2_location
    
#%% main

if __name__ == "__main__":
    data_dir =Path("C:/output_data") # NEED TO CORRECT FOR CODE OCEAN
    path = data_dir / session_id
    results_dir = data_dir / session_id / 'fib' # NEED TO CORRECT FOR CODE OCEAN

    calib_image = results_dir / 'CalibrationImage.tiff'
    calib_file = results_dir / 'calibration.txt'

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")


    num_frames, times, frames, metadata_files = load_session_params(results_dir)

    print("Processed CSV files (in order):")
    for p in metadata_files:
        print("  -", p.name)
    print("\nnumber of frames per recording fragment:", num_frames)
    for i, (t, f) in enumerate(zip(times, frames)):
        print(f"\nRecording {i}:")
        #print(f"  time: array length = {t.size}, example [-5:] = {t[-5:]}")
        print(f"  Frames (corrected): length = {f.size}, example [-5:] = {f[-5:]}")
        
    
    # Read in camera dimensions and offsets, convert to integers
    metadata = pd.read_csv(metadata_files[0])
    width = metadata.Width[0]
    height = metadata.Height[0]
    # Use XOffset if available (Bonsai node V3.2) or Left if not (V4)
    if hasattr(metadata, "XOffset"):
        Xoffset = int(metadata.XOffset[0])
    else:
        Xoffset = int(metadata.Left[0])
    # Use 'YOffset' if available, otherwise fall back to 'Top'
    if hasattr(metadata, "YOffset"):
        Yoffset = int(metadata.YOffset[0])
    else:
        Yoffset = int(metadata.Top[0])
    print('\nWidth: ' + str(width))
    print('Height: ' + str(height))
    print('X Offset: ' + str(Xoffset))
    print('Y Offset: ' + str(Yoffset))
    
    
    # load calibration.txt values
    theta_r, pt1, pt2, pt3, pt4, pt5, pt6, fiber1_location, fiber2_location = load_calib_values(calib_file, Xoffset, Yoffset)    
    
#%%