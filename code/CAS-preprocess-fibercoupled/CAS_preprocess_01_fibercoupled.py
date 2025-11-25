# CAS_preprocess_01_calibration_V2.py

# TO DO: 
# figure out best way to identify session ID for import
# currently expects CalibrationFiles inside of fib folder, not sure this is how it will actually be in preprocessed form
# V2 fixes the image 'view' vs 'copy' issue that was changing the calibration image within a loop

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from scipy.signal import find_peaks
import cv2 as cv
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# constants
SAT_VAL = 12000 # saturation value of camera
FIBER_WIDTH = 60 # width of fiber (in pixels)

def load_session_paths(data_dir, session_id):
    """Create paths to session data and calibration files."""
    path = os.path.join(data_dir, session_id)
    calib_path = os.path.join(path, 'fib', 'CalibrationFiles')
    return path, calib_path


def load_calibration_metadata(calib_path):
    """Load session_params CSV from calibration folder."""
    metadata_files = glob.glob(os.path.join(calib_path, '*.csv'))
    session_params_file = [f for f in metadata_files if 'session_params' in os.path.basename(f)]
    
    if not session_params_file:
        raise FileNotFoundError(f"No session_params CSV file found in {calib_path}")
    
    metadata = pd.read_csv(session_params_file[0])
    return metadata


def load_and_average_tiff(tiff_dir):
    """Load multi-frame TIFF and average frames into a 2D image."""
    tiff_files = np.array(natsorted(os.listdir(tiff_dir)))
    
    # Filter for TIFF files only
    tiff_files = [f for f in tiff_files if f.lower().endswith(('.tif', '.tiff'))]
    
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {tiff_dir}")
    
    tiff_file_path = os.path.join(tiff_dir, tiff_files[0])
    tiff = Image.open(tiff_file_path)

    image_sequence = []
    while True:
        try:
            image_sequence.append(np.array(tiff))
            tiff.seek(tiff.tell() + 1)
        except EOFError:
            break

    img = np.array(image_sequence)
    img2d = img.mean(axis=0)
    return img2d


def find_laser_positions(img, height_thresh=2000, dist_thresh=50, sat_val=SAT_VAL):
    """Find horizontal and vertical positions of lasers in the image."""
    img_to_unskew = np.array(img)
    h_line = np.mean(img_to_unskew, axis=0)
    h_peaks, _ = find_peaks(h_line, height=height_thresh, distance=dist_thresh)

    v_peaks = np.zeros_like(h_peaks, dtype=int)
    
    # sat_val = 12000
    for i, x in enumerate(h_peaks):
        v_line = img_to_unskew[:, x].copy()
        v_line[v_line < sat_val] = 0
        idx = np.nonzero(np.diff(v_line))[0]
        
        if len(idx) >= 2:  # Need at least 2 edges
            v_peaks[i] = int(np.round((idx[0] + idx[-1]) / 2))
        else:
            print(f"Warning: Could not find edges for laser at x={x}")
            
    return h_peaks, v_peaks, img_to_unskew


def rotate_image(img, h_peaks, v_peaks):
    """Rotate image to align lasers horizontally."""
    if len(h_peaks) < 2 or len(v_peaks) < 2:
        raise ValueError(f"Not enough laser peaks found. Found {len(h_peaks)} horizontal and {len(v_peaks)} vertical peaks.")
    
    theta_r = np.rad2deg(np.arctan(
        (v_peaks[-2] - v_peaks[0]) / (h_peaks[-2] - h_peaks[0])
    ))
    rows, cols = img.shape
    M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), theta_r, 1)
    img_rotated = cv.warpAffine(img, M, (cols, rows))
    return img_rotated, theta_r


def analyze_laser(img, h_peaks_rot, use_laser, sat_val=SAT_VAL, fiber_width=FIBER_WIDTH):
    """Find vertical and horizontal boundaries of a laser fiber."""
    # Check if use_laser index is valid
    if use_laser >= len(h_peaks_rot):
        raise IndexError(f"use_laser={use_laser} but only {len(h_peaks_rot)} lasers detected")
    
    # Vertical
    #v_line = img_copy[:, h_peaks_rot[use_laser]]
    v_line = img[:, h_peaks_rot[use_laser]].copy()
    v_line[v_line < sat_val] = 0
    idx = np.nonzero(np.diff(v_line))[0]
    
    if len(idx) < 2:
        raise ValueError(f"Could not find edges for laser {use_laser}")
        
    v_width = [int(idx[0]), int(idx[-1])]

    # Horizontal
    h_width = []
    for position in [v_width[0] + 5, v_width[1] - 5]:
        h_start = max(0, h_peaks_rot[use_laser] - fiber_width)
        h_end = min(img.shape[1], h_peaks_rot[use_laser] + fiber_width)
        h_line = img[position, h_start:h_end].copy() 
        h_line[h_line < sat_val] = 0
        h_edges = np.nonzero(np.diff(h_line))[0]
        
        if len(h_edges) < 1:
            raise ValueError(f"Could not find horizontal edges for laser {use_laser} at position {position}")
        
        h_width.append(int(h_peaks_rot[use_laser]-fiber_width + h_edges[0]))

    return v_width, h_width


def perform_affine_transform(img_rotated, points_src, points_dst):
    """Apply affine transformation to align fibers."""
    pts1 = np.float32(points_src)
    pts2 = np.float32(points_dst)
    M = cv.getAffineTransform(pts1, pts2)
    rows, cols = img_rotated.shape
    img_final = cv.warpAffine(img_rotated, M, (cols, rows))
    return img_final


def store_fiber_boundaries(img_final, h_peaks_final, use_laser, sat_val=SAT_VAL):
    """Find relative pixel locations of each fiber."""
    if use_laser >= len(h_peaks_final):
        raise IndexError(f"use_laser={use_laser} but only {len(h_peaks_final)} lasers detected in final image")
    
    v_line = img_final[:,h_peaks_final[use_laser]].copy()
    v_line[v_line < sat_val] = 0
    idx = np.nonzero(np.diff(v_line))[0]
    
    if len(idx) < 2:
        raise ValueError(f"Could not find fiber edges for laser {use_laser}")
        
    v_width = [int(idx[0]), int(idx[-1])]
    
    fiber1 = [v_width[0] + 20, v_width[0] + 40]
    fiber2 = [v_width[1] - 40, v_width[1] - 20]
    return fiber1, fiber2


def save_results(img_final, theta_r, points, fiber_bounds, Xoffset, Yoffset, results_dir):
    """Save transformed TIFF and calibration points."""
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "CalibrationImage.tiff"
    cv.imwrite(str(output_file), img_final.astype(int))
    # cv.imwrite(str(output_file), img_final.astype(uint16)) # should we use uint16 since it's a 16-bit image?

    with open(results_dir / 'calibration.txt', 'w') as f:
        f.write(f'rot_tform_thetaR = {theta_r}\n')
        for i, pt in enumerate(points, 1):
            f.write(f'aff_tform_pt{i} = {pt}\n')
        f.write(f'fiber1_pixels = {fiber_bounds[0]}\n')
        f.write(f'fiber2_pixels = {fiber_bounds[1]}\n')
        f.write(f'calib_Xoffset = {Xoffset}\n')
        # f.write(f'calib_Yoffset = {Yoffset}\n')  # Also save Yoffset
    
    print(f"Results saved to {results_dir}")
    print(f"  - CalibrationImage.tiff")
    print(f"  - calibration.txt")

    
    
if __name__ == '__main__':
    print("Starting HSFP image calibration processing...")
        
    # Settings
    data_dir = '/data/' # NEED TO CORRECT
    
    # Get session ID
    session_id = os.listdir(data_dir)[0] # NEED TO CORRECT
    path, calib_path = load_session_paths(data_dir, session_id)
    metadata = load_calibration_metadata(calib_path)
    tiff_dir = os.path.join(calib_path, 'Tiffs')
    img2d = load_and_average_tiff(tiff_dir)

    # Camera offsets
    Xoffset = int(metadata.XOffset[0])
    Yoffset = int(metadata.YOffset[0])

    # Find laser positions
    h_peaks, v_peaks, img_to_unskew = find_laser_positions(img2d)
    
    # Rotate image
    img_rotated, theta_r = rotate_image(img_to_unskew, h_peaks, v_peaks)
    
    # Calculate centers of each laser after rotation
    h_line = np.mean(img_rotated, axis=0)
    h_peaks_rot, _ = find_peaks(h_line, height=2000, distance=50)

#     v_peaks_rot = np.zeros(np.size(h_peaks_rot))
#     for i, x in enumerate(h_peaks_rot):
#         v_line = img_rotated[:, x].copy()
#         v_line[v_line < sat_val] = 0
#         idx = np.nonzero(np.diff(v_line))[0]

#         if len(idx) >= 2:  # Need at least 2 edges
#             v_peaks_rot[i] = int(np.round((idx[0] + idx[-1]) / 2))
#         else:
#             print(f"Warning: Could not find edges for laser at x={x}")
#     v_peaks_rot = v_peaks_rot.astype(int)
    
    # Analyze fibers
    v_width1, h_width1 = analyze_laser(img_rotated, h_peaks_rot, use_laser=0)
    v_width2, h_width2 = analyze_laser(img_rotated, h_peaks_rot, use_laser=2)

    # Define affine points and transform image
    pt1, pt2, pt3 = [h_width1[0], v_width1[0]], [h_width1[1], v_width1[1]], [h_width2[0], v_width2[0]]
    pt4, pt5, pt6 = [h_width1[1], v_width1[0]], [h_width1[1], v_width1[1]], [h_width2[1], v_width2[0]]

    img_final = perform_affine_transform(img_rotated, [pt1, pt2, pt3], [pt4, pt5, pt6])

    # Store fiber boundaries
    h_line_final = np.mean(img_final, axis=0)
    h_peaks_final, _ = find_peaks(h_line_final, height=2000, distance=50)
    
#     v_peaks_final = np.zeros(np.size(h_peaks_final))
#     for i, x in enumerate(h_peaks_final):
#         v_line = img_final[:, x].copy()
#         v_line[v_line < SAT_VAL] = 0
#         idx = np.nonzero(np.diff(v_line))[0]

#         if len(idx) >= 2:  # Need at least 2 edges
#             v_peaks_final[i] = int(np.round((idx[0] + idx[-1]) / 2))
#         else:
#             print(f"Warning: Could not find edges for laser at x={x}")
#     v_peaks_final = v_peaks_final.astype(int)
    
    fiber1, fiber2 = store_fiber_boundaries(img_final, h_peaks_final, use_laser=2)
    

    # Save results
    pt1[0] = int(pt1[0] + Xoffset)
    pt2[0] = int(pt2[0] + Xoffset)
    pt3[0] = int(pt3[0] + Xoffset)
    pt4[0] = int(pt4[0] + Xoffset)
    pt5[0] = int(pt5[0] + Xoffset)
    pt6[0] = int(pt6[0] + Xoffset)
    pt1[1] = int(pt1[1] + Yoffset)
    pt2[1] = int(pt2[1] + Yoffset)
    pt3[1] = int(pt3[1] + Yoffset)
    pt4[1] = int(pt4[1] + Yoffset)
    pt5[1] = int(pt5[1] + Yoffset)
    pt6[1] = int(pt6[1] + Yoffset)
    fiber1[0] = int(fiber1[0] + Yoffset)
    fiber1[1] = int(fiber1[1] + Yoffset)
    fiber2[0] = int(fiber2[0] + Yoffset)
    fiber2[1] = int(fiber2[1] + Yoffset)
    
    results_dir = Path('/results/') # NEED TO CORRECT
    save_results(img_final, theta_r, [pt1, pt2, pt3, pt4, pt5, pt6], [fiber1, fiber2], Xoffset, Yoffset, results_dir)
    
    print("Calibration processing complete.")