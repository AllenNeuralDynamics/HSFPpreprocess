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


def check_frame_drop(frames_list, times_list):
    """
    1. Detects gaps in hardware framestamps.
    2. Plots Inter-Frame Interval (IFI) for timing stability analysis.
    3. Interpolates missing metadata to maintain 1:1 TIFF alignment.
    """
    repaired_f = []
    repaired_t = []

    for i, (f_arr, t_arr) in enumerate(zip(frames_list, times_list)):
        # --- PART 1: DIAGNOSTIC PLOTTING ---
        # calculate the inter-frame-interval, convert to ms
        ifi_ms = np.diff(t_arr) * 1000
        
        # plot the ifi stability across the recording
        plt.figure(figsize=(10, 4))
        plt.plot(ifi_ms, color='#1f77b4', alpha=0.7)
        plt.axhline(y=np.median(ifi_ms), color='r', linestyle='--', 
                    label=f'Median: {np.median(ifi_ms):.2f}ms')
        
        plt.title(f"Timing Stability Raw: Segment {i} ({len(f_arr)} frames)")
        plt.xlabel("Frame Index")
        plt.ylabel("Inter-Frame Interval (ms)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # --- PART 2: DROP DETECTION & REPAIR ---
        diffs = np.diff(f_arr)
        gap_indices = np.where(diffs > 1)[0]
        
        # if no frame drops are detected:
        if len(gap_indices) == 0:
            print(f"Segment {i}: No dropped frames detected.")
            repaired_f.append(f_arr)
            repaired_t.append(t_arr)
            continue # acts as early exit to rest of the for loop

        # Repair logic if there are gaps:
        f_list = f_arr.tolist()
        t_list = t_arr.tolist()
        total_inserted = 0
        
        # if there is a frame drop, interpolate a time stamp (work in reverse
        # to prevent correction from altering other dropped frames):
        for idx in reversed(gap_indices):
            # identify the size of the gap (e.g. 1 or 2 frames dropped in a row)
            num_missing = int(diffs[idx] - 1)
            # identify the timestamps immediately before and after the drop
            t_start, t_end = t_arr[idx], t_arr[idx+1]
            # interpolate missing time stamp, then strip away 'known' start and 
            # end stamps to keep only new interpolated stamp(s)
            interp_ts = np.linspace(t_start, t_end, num_missing + 2)[1:-1]
            
            for j in range(num_missing):
                inserted_fs = f_arr[idx] + (j + 1)
                print(f"Segment {i}: Repairing gap at index {idx}. Inserting Framestamp: {inserted_fs}")
                
                # add the missing frame counter
                f_list.insert(idx + 1, inserted_fs)
                # add the newly calculated timestamp
                t_list.insert(idx + 1, interp_ts[j])
                total_inserted += 1
                
        # plot the corrected ifi stability
        ifi_ms_corr = np.diff(t_list) * 1000
        plt.figure(figsize=(10, 4))
        plt.plot(ifi_ms_corr, color='#1f77b4', alpha=0.7)
        plt.axhline(y=np.median(ifi_ms_corr), color='r', linestyle='--', 
                     label=f'Median: {np.median(ifi_ms_corr):.2f}ms')
        plt.title(f"Timing Stability CORRECTED: Segment {i} ({len(f_list)} frames)")
        plt.xlabel("Frame Index")
        plt.ylabel("Inter-Frame Interval (ms)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        
        # append corrected frame and time series for current recording segment to the list
        print(f"Segment {i}: Detected {len(gap_indices)} gaps. Inserted {total_inserted} frames.")
        repaired_f.append(np.array(f_list))
        repaired_t.append(np.array(t_list))
        
        
    return repaired_f, repaired_t



    
    

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