# ASH Neuronal Calcium Imaging Analysis Pipeline

This folder contains scripts for analyzing ASH neuron calcium imaging data using ratiometric GCaMP/RFP recordings.

## Overview

The pipeline processes dual-channel (GCaMP and RFP) calcium imaging videos to extract ASH neuron activity. It uses TMAC (Two-channel Motion Artifact Correction) to correct for photobleaching and motion artifacts, then analyzes activity traces aligned to behavioral events.

## Required Packages

- Python 3.x
- numpy
- opencv-cv2 (cv2)
- tifffile
- pandas
- matplotlib
- imageio
- pickle
- tmac (for TMAC_analysis.py) https://github.com/Nondairy-Creamer/tmac
- scipy

## Pipeline Order

### Main Analysis Pipeline

Run scripts in this order:

1. **Convert TIF to Stack.py** - Convert individual TIFF images to a single stack
2. **ASH_Segmentation.py** - Segment ASH neurons from the image stack (run for both channels)
3. **CSV_formatting.py** - Combine red and green channel data
4. **TMAC_analysis.py** - Apply TMAC correction for photobleaching and motion artifacts
5. **PickletoCSV.py** - Convert TMAC output (.pkl) to CSV format
6. **Plotting.py** - Visualize calcium traces

### Activity Trace Alignment Sub-Pipeline

For aligning activity traces to behavioral events (e.g., reversal onset):

Navigate to `TQ5856/ActivityTraces/` and run:

1. **Align_timeseries.py** - Align time series data to specified time points
2. **SummaryPlot.py** - Generate summary plots with mean ± SEM

---

## Detailed Script Documentation

### 1. Convert TIF to Stack.py

**Purpose:** Converts a folder of individual TIFF images into a single multi-frame TIFF stack.

**Input:**
- Directory containing individual `.tiff` files (numbered sequentially)

**Output:**
- Single multi-frame TIFF stack file

**Configuration Required:**
```python
directory = r'path/to/tiff/folder'  # Folder containing individual TIFF files
name = 'GcaMP_stack'                # Output filename (without extension)
```

**Notes:**
- Files are sorted numerically before stacking
- Run this separately for GCaMP and RFP channels
- Output is saved in the parent directory of the input folder

---

### 2. ASH_Segmentation.py

**Purpose:** Automatically segments ASH neuron soma from calcium imaging stacks using adaptive thresholding and contour detection.

**Input:**
- TIFF stack from step 1

**Output:**
- CSV file with columns: `frame`, `25px sum`, `25px mean`, `10px sum`, `10px mean`
- GIF visualization showing segmentation quality

**Configuration Required:**
```python
threshold = 7           # Adaptive threshold block size (increase for less selective)
subtraction = -7        # Threshold constant (more negative = more selective)
directory = "/path/to/WormsPy/output/"
name = "GCaMP_stack"    # Base name for input/output files
```

**Notes:**
- Uses adaptive Gaussian thresholding with morphological operations
- Tracks centroid across frames (max distance threshold = 10 pixels)
- Extracts brightest 25 and 10 pixels within 10-pixel radius ROI
- Visual feedback: displays real-time segmentation with contours and ROI
- Run separately for each channel (GCaMP and RFP)

---

### 3. CSV_formatting.py

**Purpose:** Combines GCaMP and RFP channel data into a single CSV file.

**Input:**
- Two CSV files from ASH_Segmentation.py (one per channel)

**Output:**
- Combined CSV with columns: `RFP_raw`, `GCaMP_raw`

**Configuration Required:**
```python
df1 = pd.read_csv('path/to/wormXR.csv')  # RFP channel
df2 = pd.read_csv('path/to/wormXG.csv')  # GCaMP channel
# Output path:
combined_df.to_csv('path/to/wormX.csv', index=False)
```

**Notes:**
- Extracts the '25px_sum' column from each channel
- Replaces zero values with NaN for proper handling in TMAC
- Column names must be exactly `RFP_raw` and `GCaMP_raw`

---

### 4. TMAC_analysis.py

**Purpose:** Applies Two-channel Motion Artifact Correction (TMAC) to separate calcium activity from motion artifacts and photobleaching.

**Input:**
- Combined CSV file from step 3 with `RFP_raw` and `GCaMP_raw` columns

**Output:**
- `.pkl` file containing TMAC results (activity, motion, corrected signals)
- `.mat` file (MATLAB format)

**Configuration Required:**
```python
folder_path = 'path/to/data/folder'
CSV_File = folder_path / 'wormX.csv'
tmac_save_path = folder_path / 'wormX'  # Output base name (no extension)
sample_rate = 10  # Sampling rate in Hz
```

**Output Variables in .pkl file:**
- `a`: Calcium activity signal (corrected)
- `a_nan`: Activity with original NaN positions preserved
- `m`: Motion artifact signal
- `g_raw`, `r_raw`: Original raw signals
- `g_corrected`, `r_corrected`: Photobleaching-corrected signals
- `length_scale_a`, `length_scale_m`: GP hyperparameters
- `variance_a`, `variance_m`: Signal variances
- `variance_g_noise`, `variance_r_noise`: Noise variances

**Notes:**
- Interpolates over NaN values before processing
- Automatically corrects for exponential photobleaching
- Uses Gaussian Process regression to separate activity from motion

---

### 5. PickletoCSV.py

**Purpose:** Converts TMAC .pkl output files to CSV format for easier analysis.

**Input:**
- `.pkl` files from TMAC_analysis.py

**Output:**
- CSV files with columns: `a` (activity), `time` (in seconds)

**Configuration Required:**
```python
input_directory = 'path/to/pkl/files'
output_directory = 'path/to/output/csvs'
```

**Notes:**
- Extracts activity trace `a` from pickle file
- Calculates time assuming 10 Hz sampling rate
- Processes all .pkl files in the input directory

---

### 6. Plotting.py

**Purpose:** Visualizes calcium traces and TMAC decomposition for quality control.

**Input:**
- Single `.pkl` file from TMAC_analysis.py

**Output:**
- Interactive matplotlib plot showing activity and motion traces

**Configuration Required:**
```python
data = load_data('path/to/wormX.pkl')
```

**Plot Components:**
- Blue line: Calcium activity (`a`)
- Orange line: Motion artifacts (`m`)
- Optional: Raw and corrected GCaMP/RFP traces (uncomment in code)

---

## Activity Trace Alignment Sub-Pipeline

Location: `TQ5856/ActivityTraces/`

### 7. Align_timeseries.py

**Purpose:** Aligns multiple time series to a common reference point (e.g., behavioral event onset).

**Input:**
- Multiple CSV files with `time` and `a` (activity) columns
- Alignment time points for each file

**Output:**
- CSV files with suffix `_aligned` containing additional `aligned_time` column

**Usage:**
```bash
python Align_timeseries.py --files worm1.csv worm3.csv worm6.csv \
                           --align_points 425 380 510 \
                           --time_col time \
                           --output_suffix _aligned
```

**Arguments:**
- `--files`: Space-separated list of CSV files to align
- `--align_points`: Space-separated list of time points (one per file) to set as t=0
- `--time_col`: Name of time column (default: "time")
- `--output_suffix`: Suffix for output files (default: "_aligned")

**Notes:**
- Number of files must match number of alignment points
- Alignment points are divided by 10 internally (assuming frame numbers)
- Original time column is preserved

---

### 8. SummaryPlot.py

**Purpose:** Creates publication-quality summary plots of aligned calcium traces.

**Input:**
- Aligned CSV files from Align_timeseries.py (in `aligned/` subfolder)

**Output:**
- `ASH_reversal_activity.pdf` and `.png` - Publication-quality figure
- Shows mean ± SEM with individual traces optional

**Configuration Required:**
```python
directory = 'aligned'      # Folder with aligned CSV files
window_size = 1           # Smoothing window size (frames)
```

**Plot Features:**
- Truncates traces to -100 to +50 frames around alignment point
- Calculates and displays mean trace with SEM shading
- Adds vertical line at t=0 (event onset)
- Automatic peak annotation
- Publication-ready formatting (300 DPI, custom fonts)

**Notes:**
- All traces are truncated to minimum common length
- Smoothing applied with rolling window (adjustable)
- Requires aligned CSV files with `aligned_time` and `a` columns

---

## Data Organization

### Input Data Structure
```
ASH/
├── Convert TIF to Stack.py
├── ASH_Segmentation.py
├── CSV_formatting.py
├── TMAC_analysis.py
├── PickletoCSV.py
├── Plotting.py
└── TQ5856/                    # Strain name
    ├── worm1G.csv             # GCaMP segmented data
    ├── worm1R.csv             # RFP segmented data
    ├── worm1.csv              # Combined data
    ├── worm1.pkl              # TMAC output
    └── ActivityTraces/
        ├── Align_timeseries.py
        ├── SummaryPlot.py
        ├── worm1.csv          # Activity traces
        └── aligned/
            └── worm1_aligned.csv
```

### Output Data Structure
- Raw segmentation: `wormXG.csv`, `wormXR.csv`
- Combined: `wormX.csv`
- TMAC outputs: `wormX.pkl`, `wormX.mat`
- Activity traces: `ActivityTraces/wormX.csv`
- Aligned traces: `ActivityTraces/aligned/wormX_aligned.csv`

---

## Important Notes

1. **Directory Paths:** All scripts require manual specification of input/output directory paths
2. **Channel Naming:** Maintain consistent naming: `wormXG.csv` (GCaMP), `wormXR.csv` (RFP), `wormX.csv` (combined)
3. **Segmentation Quality:** Check the output GIF from ASH_Segmentation.py to verify proper neuron tracking
4. **Threshold Tuning:** Adjust `threshold` and `subtraction` parameters in ASH_Segmentation.py if segmentation fails
5. **TMAC Requirements:** Requires both channels with matching lengths and no leading/trailing zeros
6. **Sampling Rate:** Default is 10 Hz - adjust in TMAC_analysis.py if different

---

## Troubleshooting

### Segmentation Issues
- **No ROI detected:** Increase `threshold` value (less selective)
- **Multiple ROIs detected:** Decrease `subtraction` value (more selective, more negative)
- **Tracking lost:** Check `distance_threshold` parameter (default: 10 pixels)

### TMAC Issues
- **Interpolation errors:** Check for NaN values in input CSV
- **Poor separation:** Ensure adequate signal-to-noise ratio in both channels
- **Memory errors:** Reduce video length or close other applications

### Alignment Issues
- **Length mismatch:** Align_timeseries requires equal-length alignment point lists
- **No aligned_time column:** Check that Align_timeseries.py completed successfully
- **Plotting errors:** Ensure all CSV files have consistent column names

---

## Citation

If using this pipeline, please cite the TMAC paper:
- Correcting motion induced fluorescence artifacts in two-channel neural imaging
Creamer MS, Chen KS, Leifer AM, Pillow JW (2022) Correcting motion induced fluorescence artifacts in two-channel neural imaging. PLOS Computational Biology 18(9): e1010421. https://doi.org/10.1371/journal.pcbi.1010421

If using this pipeline for muscle analysis, please cite:
- A novel epifluorescence microscope design and software package to record naturalistic behaviour and cell activity in freely moving Caenorhabditis elegans
Sebastian N. Wittekindt, Hannah Owens, Lennard Wittekindt, Aurélie Guisnet, Michael Hendricks
bioRxiv 2025.03.21.644605; doi: https://doi.org/10.1101/2025.03.21.644605

---

## Contact

For questions about this pipeline, contact sebastian.wittekindt@gmail.com.
