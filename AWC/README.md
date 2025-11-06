# AWC Neuronal Calcium Imaging Analysis Pipeline

This folder contains scripts for analyzing AWC neuron calcium imaging data from WormsPy recordings, integrating fluorescence segmentation with behavioral annotation using DeepLabCut (DLC).

## Overview

The pipeline processes single-channel calcium imaging videos (GCaMP) to extract neuron activity while simultaneously tracking worm behavior. It combines automated segmentation with stage position tracking and DLC-based nose-tip tracking.

## Required Packages

- Python 3.x
- numpy
- opencv-cv2 (cv2)
- tifffile
- pandas
- matplotlib
- PIL (Python Imaging Library)
- csv
- os
- re

## Pipeline Overview

This pipeline can be run either using:
1. **Jupyter Notebook** (`WormsPy_dataproc.ipynb`) - Interactive, step-by-step execution
2. **Python Script** (`Segmentation.py`) - Standalone segmentation only

We recommend using the Jupyter Notebook for the complete workflow.

---

## Complete Workflow (Jupyter Notebook)

### Required Inputs

From WormsPy recording:
- Folder of TIFF images (fluorescence channel)
- Stage position CSV (`_stagepos.csv`)
- Brightfield AVI video
- DeepLabCut output CSV (tracking nose tip)

### Workflow Steps

The Jupyter Notebook (`WormsPy_dataproc.ipynb`) contains 6 main code blocks:

#### Block 1: Format DLC CSV Files
Pre-processes DeepLabCut output files to remove artifacts and smooth tracking data.

**Input:**
- Raw DLC CSV files in specified folder

**Output:**
- Cleaned DLC CSV files with columns: `index`, `x`, `y`

**Configuration Required:**
```python
folder_path = 'DLC_FOLDER'  # Path to folder with DLC CSV files
```

**Processing Steps:**
- Removes header rows
- Filters jumps > 30 pixels (likely tracking errors)
- Applies 3-frame rolling average smoothing
- Rounds coordinates to nearest pixel

---

#### Block 2: Master Imports and Directories
Set all file paths and parameters for the recording.

**Configuration Required:**
```python
directory = "DIRECTORY"           # WormsPy output folder
name = "NAME"                     # Recording name
# Resolution parameters (update if needed):
midpoint_x = 960                  # Video width / 2
midpoint_y = 600                  # Video height / 2
pxtouM = 1.54                     # Pixel to micron conversion factor
```

**File Paths Configured:**
- `stagepos_csv`: Stage position recording
- `annotatedcsv`: Output file with behavioral annotations
- `segmented_csv_path`: Fluorescence segmentation output
- `DLC_file`: DeepLabCut nose tracking
- `nosepos_csv`: Combined nose position output

---

#### Block 3: TIFF Stacker
Converts folder of TIFF images into a single stack for segmentation.

**Input:**
- Subfolder(s) within `directory` containing individual `.tiff` files

**Output:**
- `NAME.tiff` - Multi-frame TIFF stack in `directory`

**Notes:**
- Automatically finds and processes all subdirectories
- Files are sorted numerically before stacking
- Run this before segmentation script

---

#### Block 4: Auto-Annotate Reversals
Calculates curvature and angle metrics to identify potential reversal events.

**Input:**
- Stage position CSV (`_stagepos.csv`)

**Output:**
- `NAME_annotated.csv` with additional columns:
  - `Curvatures`: Three-point curvature at each timepoint
  - `Angles`: Vertex angle between consecutive positions
  - `Reversals`: Binary flag (1 = likely reversal, 0 = forward movement)

**Configuration Required:**
```python
angle_threshold = 30        # Maximum angle for reversal (degrees)
curvature_threshold = 0.03  # Minimum curvature for reversal
```

**Notes:**
- Displays plot with detected events (blue = high angle, red = high curvature)
- Manual verification recommended - may include false positives
- For more accurate behavioral annotation, use DeepLabCut trained on your data

---

#### Block 5: Nose Tip Translation
Translates DLC pixel coordinates to absolute stage position in microns.

**Input:**
- DLC CSV (formatted in Block 1)
- Annotated stage position CSV (from Block 4)

**Output:**
- `NAME_nosecoords.csv` with added columns:
  - `X_nose`: Absolute X position of nose in microns
  - `Y_nose`: Absolute Y position of nose in microns

**Configuration Required (if needed):**
```python
midpoint_x = 960    # Video center X
midpoint_y = 600    # Video center Y
pxtouM = 1.54       # Pixel to micron conversion
```

**Calculation:**
```
X_nose = X_motor + (DLC_x - midpoint_x) * pxtouM
Y_nose = Y_motor - (DLC_y - midpoint_y) * pxtouM
```

**Notes:**
- DLC coordinates are relative to video center
- Stage coordinates are in absolute microns
- Y-axis is inverted (image coordinates vs. stage coordinates)

---

#### Block 6: Combine Data Files
Merges segmentation data with position/behavioral data.

**Input:**
- `NAME_segmented.csv` (from Segmentation.py - see below)
- `NAME_nosecoords.csv` (from Block 5)

**Output:**
- `NAME_combined.csv` - Complete dataset with:
  - Fluorescence measurements (from segmentation)
  - Stage position
  - Nose tip position
  - Curvatures and angles
  - Reversal annotations

**Notes:**
- Row alignment is automatic (based on frame number)
- This is the final output for downstream analysis

---

## Segmentation Script (Run Between Blocks 3 and 6)

### Segmentation.py

**Purpose:** Segments AWC neuron from TIFF stack using adaptive thresholding.

**Input:**
- TIFF stack created in Notebook Block 3

**Output:**
- CSV file with columns: `frame`, `25px sum`, `25px mean`, `10px sum`, `10px mean`
- GIF visualization showing segmentation quality

**Configuration Required:**
```python
threshold = 7           # Adaptive threshold block size (increase for less selective)
subtraction = -7        # Threshold constant (more negative = more selective)
directory = "DIRECTORY" # WormsPy output folder
name = "NAME"          # Base name (same as notebook)
```

**Algorithm:**
1. Frame-by-frame adaptive Gaussian thresholding
2. Morphological operations (opening → closing)
3. Contour detection with area filtering (1-40 pixels)
4. Centroid tracking (max jump = 10 pixels)
5. Extraction of brightest 25 and 10 pixels in 10-pixel radius ROI

**Notes:**
- Identical to ASH segmentation script
- Adjust `threshold` and `subtraction` if segmentation fails
- Check output GIF for quality control
- Must be run after Block 3 and before Block 6

---

## Complete Pipeline Order

1. **Run Jupyter Notebook Block 1:** Format DLC CSV files
2. **Run Jupyter Notebook Block 2:** Set directories and parameters
3. **Run Jupyter Notebook Block 3:** Create TIFF stack
4. **Run Segmentation.py:** Segment fluorescence (separate script)
5. **Run Jupyter Notebook Block 4:** Auto-annotate behavioral events
6. **Run Jupyter Notebook Block 5:** Calculate absolute nose position
7. **Run Jupyter Notebook Block 6:** Combine all data into final CSV

---

## Data Organization

### Input Structure
```
AWC/
├── Segmentation.py
├── WormsPy_dataproc.ipynb
├── DLC_FOLDER/
│   └── [DLC CSV files]
└── MMH214/                    # Strain/experiment name
    ├── BLUE_1.csv             # Individual worm data files
    ├── LAWN1_stagepos.csv     # Stage position recording
    ├── LAWN1_annotated.csv    # Annotated positions
    ├── LAWN1_segmented.csv    # Fluorescence segmentation
    └── LAWN1_combined.csv     # Final combined dataset
```

### Output Columns in Final CSV

From Segmentation:
- `frame`: Frame number
- `25px sum`, `25px mean`: ROI fluorescence (25 brightest pixels)
- `10px sum`, `10px mean`: ROI fluorescence (10 brightest pixels)

From Stage Position:
- `X_motor`, `Y_motor`: Stage position in microns
- `Curvatures`: Three-point path curvature
- `Angles`: Vertex angle in degrees
- `Reversals`: Binary reversal flag

From DLC Integration:
- `X_nose`, `Y_nose`: Absolute nose position in microns

---

## Important Notes

### Configuration Requirements
1. **Directory Paths:** Must be specified in both notebook Block 2 and Segmentation.py
2. **Name Consistency:** Use same `name` variable across notebook and segmentation script
3. **Resolution Parameters:** Verify `midpoint_x`, `midpoint_y`, and `pxtouM` match your setup

### DLC Integration
- DLC model should be trained to track nose tip reliably
- CSV formatting (Block 1) removes tracking artifacts but manual inspection recommended
- For best results, train DLC on multiple behavioral states (forward, reversal, omega turns)

### Behavioral Annotation
- Auto-annotation (Block 4) provides starting point but requires validation
- Adjust `angle_threshold` and `curvature_threshold` for your data
- Consider training custom DLC model for more accurate behavioral classification

### Segmentation Quality Control
- Always check the output GIF from Segmentation.py
- Green contours should tightly follow neuron boundary
- Red circle should center on neuron centroid
- If tracking is lost, tune `threshold` and `subtraction` parameters

---

## Troubleshooting

### DLC Formatting Issues
- **Large jumps in tracking:** Lower threshold in Block 1 (currently 30 pixels)
- **Excessive smoothing:** Reduce rolling window size (currently 3 frames)

### Segmentation Issues
- **No ROI detected:** Increase `threshold` value (less selective thresholding)
- **Multiple ROIs detected:** Decrease `subtraction` value (more negative, more selective)
- **Centroid jumps:** Check `distance_threshold` parameter (default: 10 pixels)

### Data Combination Issues
- **Length mismatch:** Ensure segmentation ran on complete TIFF stack
- **Missing columns:** Verify all notebook blocks ran successfully before Block 6
- **Coordinate transform errors:** Check `pxtouM`, `midpoint_x`, `midpoint_y` values

### Notebook Execution
- **Import errors:** Ensure all packages installed in active environment
- **File not found:** Use absolute paths or verify working directory
- **Memory errors:** Process shorter videos or reduce image resolution

---

## Advanced Usage

### Custom DLC Models
To use custom DLC models for behavioral classification:
1. Train DLC on your specific behaviors
2. Update `folder_path` in Block 1
3. Modify Block 5 if tracking different body parts
4. Adjust coordinate transformation as needed

### Multi-Neuron Tracking
To track multiple neurons:
1. Modify Segmentation.py to track multiple ROIs
2. Update CSV column names accordingly
3. Modify Block 6 to handle additional fluorescence columns

### Alternative Behavioral Metrics
To calculate custom behavioral metrics:
1. Add calculations to Block 4
2. Update `annotatedcsv` column names
3. Ensure new columns propagate through to final output

---

## Output Data Usage

The final combined CSV (`NAME_combined.csv`) contains all information needed for:
- Calcium signal analysis
- Behavioral correlations
- Trajectory analysis
- Multi-modal data visualization
- Statistical analyses linking neural activity to behavior

---

## Citation

If using this pipeline for muscle analysis, please cite:
- A novel epifluorescence microscope design and software package to record naturalistic behaviour and cell activity in freely moving Caenorhabditis elegans
Sebastian N. Wittekindt, Hannah Owens, Lennard Wittekindt, Aurélie Guisnet, Michael Hendricks
bioRxiv 2025.03.21.644605; doi: https://doi.org/10.1101/2025.03.21.644605

---

## Contact

For questions about this pipeline, contact sebastian.wittekindt@gmail.com.
