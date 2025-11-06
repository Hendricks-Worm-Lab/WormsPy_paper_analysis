# RIA Interneuron Calcium Imaging Analysis Pipeline

This folder contains scripts for analyzing RIA interneuron calcium activity using SAM2-based segmentation and polar coordinate transformation for behavioral correlation analysis.

## Overview

The pipeline processes calcium imaging videos of worms expressing RIA::GCaMP, semi-automatically segments RIA axonal compartments using SAM2 (Segment Anything Model 2), tracks head angle dynamics, and transforms neural activity into polar coordinate space for analysis of neural activity patterns relative to head movement state.

## Required Packages

- Python 3.x
- torch (PyTorch with CUDA/MPS support recommended)
- sam2 (Segment Anything Model 2) 
- opencv-cv2 (cv2)
- tifffile
- numpy
- pandas
- matplotlib
- h5py
- PIL (Python Imaging Library)
- scikit-image
- tqdm
- scipy

**SAM2 Installation:**
```bash
# Follow installation instructions at:
# https://github.com/facebookresearch/segment-anything-2
```

---

## Pipeline Order

Run scripts in numerical order (indicated by filename prefix):

1. **1ConvertTIFFtoStack.py** - Convert individual TIFFs to stacks
2. **2TIFF to JPG.py** - Convert TIFF stacks to JPG sequences
3. **3AutoCrop.py** - Auto-crop videos to RIA region (SAM2-based)
4. **4RIAMaskGen.py** - Generate RIA segmentation masks (SAM2-based)
5. **5Midline_Skeletonize.py** - Extract worm midline and head angle
6. **6smooth.py** - Smooth angle time series
7. **7Convert_polar.py** - Transform to polar coordinates

---

## Script Documentation

### 1. 1ConvertTIFFtoStack.py

**Purpose:** Converts folders of individual TIFF images into multi-frame TIFF stacks.

**Input:**
- Base directory containing multiple subfolders, each with sequential `.tif` files

**Output:**
- One TIFF stack per subfolder: `[foldername]_stack.tiff`

**Configuration Required:**
```python
base_directory = r"PATH/TO/BASE/DIRECTORY"  # Line 46
```

**Features:**
- Processes all subdirectories automatically
- Progress bars with tqdm
- Numerical sorting of files
- Preserves original data type

**Notes:**
- Output stacks saved in base directory (not in subfolders)
- Files must have `.tif` extension (not `.tiff`)

---

### 2. 2TIFF to JPG.py

**Purpose:** Converts TIFF stacks to JPG image sequences for SAM2 processing.

**Input:**
- Directory containing `.tif` or `.tiff` stack files

**Output:**
- For each stack: subfolder with JPG frames (`000000.jpg`, `000001.jpg`, ...)
- Metadata file: `conversion_info.txt`

**Configuration Required:**
```python
input_directory = "1TIFF"           # Directory with TIFF stacks (line 126)
output_directory = "2JPG"           # Output directory for JPG sequences (line 127)
normalization_method = 'minmax'     # Normalization: 'minmax', 'percentile', or 'global' (line 133)
jpg_quality = 100                   # JPG quality 0-100 (line 135)
```

**Normalization Methods:**
- **`minmax`** (recommended): Per-frame min-max normalization
  - Preserves frame-to-frame variation
  - Each frame uses full dynamic range
- **`percentile`**: Per-frame percentile-based (1%-99%)
  - Robust to outliers
  - Clips extreme values
- **`global`**: Global min-max across entire stack
  - Maintains relative intensities across frames
  - May lose detail if dynamic range varies

**Features:**
- Handles 2D, 3D, and 4D stacks (multi-channel)
- Automatic multi-channel to grayscale conversion
- Progress tracking
- Saves conversion metadata

**Notes:**
- JPG compression is lossy but speeds up SAM2 processing
- Use `quality=100` to minimize artifacts
- Creates subfolders automatically

---

### 3. 3AutoCrop.py

**Purpose:** Automatically crops videos to RIA region using SAM2-based segmentation and tracking.

**Input:**
- Directory of JPG sequences (from step 2) OR TIFF stacks (from step 1)

**Output:**
- Cropped TIFF stacks OR JPG sequences (110×110 pixels, fixed crop)
- Mask overlays for quality control

**Configuration Required:**
```python
# Line 311-313:
parent_video_dir = '2JPG'      # Input JPG sequences
crop_dir = '3CROP'             # Output cropped videos
tiff_dir = '1TIFF'            # Input TIFF stacks (alternative)

# SAM2 model paths (lines 16-18):
sam2_checkpoint = r"path/to/sam2.1_hiera_tiny.pt"
model_cfg = r"path/to/sam2.1_hiera_t.yaml"

# Processing parameters (lines 21-24):
CONFIDENCE_THRESHOLD = 0.9    # Minimum tracking confidence
QUALITY_CHECK_INTERVAL = 100  # Check quality every N frames
VIS_INTERVAL = 100            # Visualize every N frames
CHUNK_SIZE = 300              # Frames per processing chunk
OVERLAP = 1                   # Frame overlap between chunks
```

**Interactive Prompting:**
1. On first frame, draw bounding box around RIA region
2. Press number key `1` to select RIA object
3. Click and drag to draw box
4. Press `u` to undo, `c` to clear, `enter` to finish
5. SAM2 propagates mask through video automatically

**Quality Control Features:**
- Periodic confidence checking
- Auto-reprompting if tracking quality degrades
- Visualization every N frames
- User can provide corrections during processing

**Chunking System:**
- Processes long videos in chunks to avoid memory issues
- Automatically handles chunk boundaries with overlap
- Auto-seeds subsequent chunks from previous masks

**Notes:**
- GPU/CUDA highly recommended for speed
- First video requires manual prompt; subsequent videos auto-seed
- Cropped videos are 110×110 pixels centered on RIA
- Processes all unprocessed videos in alphabetical order

---

### 4. 4RIAMaskGen.py

**Purpose:** Generates detailed RIA segmentation masks using SAM2 for precise intensity extraction.

**Input:**
- Cropped JPG sequences (from step 3)

**Output:**
- HDF5 files (`.h5`) with boolean masks per frame and object
- Overlay videos showing mask quality (`[name]_overlay.mp4`)

**Configuration Required:**
```python
# Line 291-293:
crop_videos_dir = '4CROP_JPG'          # Input cropped videos
segmented_videos_dir = '5RIA_SEGMENT'   # Output directory
output_dir = '5RIA_SEGMENT'

# SAM2 model paths (lines 17-19):
sam2_checkpoint = r"path/to/sam2.1_hiera_base_plus.pt"
model_cfg = r"path/to/sam2.1_hiera_b+.yaml"

# Processing parameters (lines 22-26):
CHUNK_SIZE = 200                # Reduced to prevent memory issues
OVERLAP = 1                     # Frame overlap
VIS_INTERVAL = 10              # Visualize every N frames
CONFIDENCE_THRESHOLD = 0.8      # Minimum confidence
QUALITY_CHECK_INTERVAL = 10     # Check every N frames
```

**Interactive Prompting:**
1. Draw bounding boxes for two objects: nrD (dorsal) and nrV (ventral)
2. Press `1` for nrD, `2` for nrV
3. Click and drag to draw bounding boxes
4. SAM2 segments and tracks both neurons

**Quality Control:**
- Automatic confidence monitoring
- Re-prompting if tracking degrades
- Mask overlap detection
- Distance checking between nrD and nrV
- Post-processing analysis of all frames

**HDF5 Structure:**
```
video_name.h5
├── masks/
│   ├── 1/          # nrD masks [n_frames, 1, H, W]
│   └── 2/          # nrV masks [n_frames, 1, H, W]
└── attrs
    ├── num_frames
    ├── object_ids
    └── mask_dimensions
```

**Analysis Output:**
- Empty masks count
- Large masks (>800 pixels)
- Overlapping masks
- Distant masks
- Quality statistics

**Notes:**
- Processes all videos alphabetically
- Can skip videos by closing prompt window without drawing
- Generates overlay video for visual QC
- Color coding: Red = nrD, Blue = nrV

---

### 5. 5Midline_Skeletonize.py

**Purpose:** Extracts worm midline, calculates head angles, and RIA position angles for behavioral correlation.

**Input:**
- Brightfield AVI video

**Output:**
- CSV file: `[name]_angle.csv` with columns:
  - `frame`: Frame number
  - `RIA_angle`: Angle of RIA segment (0-1 along body)
  - `head_angle`: Angle at head region (based on points 0-2-3)
- Annotated video: `[name]_skeleton.avi`

**Configuration Required:**
```python
# Lines 133-139:
name = 10                      # Video/worm identifier
HEAD_POSITION = "BOTTOM"       # Head detection: "LEFT", "RIGHT", "TOP", "BOTTOM", "AUTO", "AUTO_ALT"
in_folder = 'RIA_sinusoidal'  # Input video folder
input_filename = os.path.join(in_folder, str(name) + ".avi")
NUM_POINTS = 12               # Points sampled along midline
out_folder = 'angles'         # Output folder
```

**Head Position Options:**
- **`LEFT`**: Leftmost point (min X)
- **`RIGHT`**: Rightmost point (max X)
- **`TOP`**: Topmost point (min Y)
- **`BOTTOM`**: Bottommost point (max Y)
- **`AUTO`**: Farthest from centroid (default)
- **`AUTO_ALT`**: Closest to centroid

**Algorithm:**
1. **Segmentation:**
   - Gaussian blur (51×51 kernel) for smoothing
   - Otsu thresholding with inversion
   - Morphological operations (erosion, dilation)
   
2. **Skeletonization:**
   - Binary skeleton extraction
   - Longest contour selection
   - Head-to-tail ordering
   
3. **Angle Calculation:**
   - **RIA_angle**: Direction of segment 0→1 (head segment)
   - **head_angle**: Angle between vectors 3→2 and 2→0
   - Uses `atan2` for proper quadrant handling

**Visual Outputs:**
- Red circle: Head position
- Blue circle: Tail position
- Yellow line segments: Body skeleton
- Orange circles: Sampled points (numbered)
- Magenta lines: Vectors for head angle calculation
- Green text: Head angle value

**Notes:**
- Adjust `HEAD_POSITION` if automatic detection fails
- `NUM_POINTS` affects spatial resolution (more points = finer detail)
- Output angles in degrees, relative to horizontal (0° = right)

---

### 6. 6smooth.py

**Purpose:** Applies moving average smoothing to angle time series to reduce noise.

**Input:**
- CSV from step 5 with columns: `frame`, `RIA_angle`, `head_angle`

**Output:**
- Smoothed CSV: `[name]_angle_smooth.csv`

**Configuration Required:**
```python
folder = 'angles'               # Input/output folder
name = '10_angle'              # Base filename (without .csv)
INPUT_FILE = os.path.join(folder, name + ".csv")
OUTPUT_FILE = os.path.join(folder, name + "_smooth.csv")
WINDOW_SIZE = 5                # Smoothing window size (frames)
```

**Algorithm:**
- Rolling window average (centered)
- Edge handling: forward/backward fill for NaN values
- Applied to both `RIA_angle` and `head_angle` columns

**Window Size Recommendations:**
- **5 frames** (default): Balances noise reduction and responsiveness
- **3 frames**: Minimal smoothing, preserves rapid changes
- **7-10 frames**: Stronger smoothing for noisy data

**Notes:**
- Does not alter frame count or timing
- Preserves original column structure
- Use for visualization and analysis, keep raw data for verification

---

### 7. 7Convert_polar.py

**Purpose:** Transforms head angle dynamics into polar coordinate space for neural state analysis.

**Input:**
- Smoothed CSV from step 6

**Output:**
- Polar coordinates CSV: `[name]_polar.csv` with columns:
  - `frame`, `head_angle`, `dhead_dt`: Original data
  - `theta_rad`, `theta_deg`: Polar angle in radians/degrees
  - `bin_index`: 10-degree bin assignment (0-35)
  - `binned_theta_deg`, `binned_theta_rad`: Binned polar angle
  - `polar_x`, `polar_y`: Unit circle coordinates
- Visualization plots (4 subplots)

**Configuration Required:**
```python
input_path = "angles/1_angle_smooth.csv"   # Input file (line 77)
output_path = "angles/1_polar.csv"          # Output file (line 78)
```

**Algorithm:**
1. Calculate derivative: `dhead/dt = gradient(head_angle)`
2. Convert to polar angle: `θ = atan2(dhead/dt, head_angle)`
3. Normalize to [0, 2π]
4. Bin into 10-degree segments (36 bins total)
5. Map to unit circle: `x = cos(θ)`, `y = sin(θ)`

**Visualization Outputs:**
1. **Original head angle** time series
2. **Phase space**: head_angle vs. dhead/dt (colored by time)
3. **Unit circle**: Binned polar coordinates
4. **Binned angles**: Temporal evolution of binned angles

**Interpretation:**
- Polar angle represents combined state of head position and movement
- Clustering in polar space indicates repeated behavioral motifs
- Transitions between clusters indicate state changes

**Notes:**
- 10-degree binning reduces dimensionality while preserving structure
- Color maps show temporal progression
- Useful for correlating neural activity with behavioral states

---

## Complete Workflow Example

### Processing a New Dataset

```bash
# 1. Convert TIFFs to stacks
python 1ConvertTIFFtoStack.py
# Configure: base_directory

# 2. Convert stacks to JPG
python "2TIFF to JPG.py"
# Configure: input_directory, output_directory

# 3. Auto-crop to RIA region
python 3AutoCrop.py
# Interactive: Draw bounding box on first frame
# Waits for: SAM2 model download if first time

# 4. Generate RIA masks
python 4RIAMaskGen.py
# Interactive: Draw boxes for nrD (1) and nrV (2)
# Output: HDF5 masks and overlay videos

# 5. Extract head angles (requires brightfield video)
python 5Midline_Skeletonize.py
# Configure: name, HEAD_POSITION, input paths

# 6. Smooth angles
python 6smooth.py
# Configure: name, WINDOW_SIZE

# 7. Convert to polar coordinates
python 7Convert_polar.py
# Configure: input_path, output_path
```

---

## Data Organization

### Directory Structure
```
RIA/
├── 1ConvertTIFFtoStack.py
├── 2TIFF to JPG.py
├── 3AutoCrop.py
├── 4RIAMaskGen.py
├── 5Midline_Skeletonize.py
├── 6smooth.py
├── 7Convert_polar.py
├── 1TIFF/                        # Input: TIFF stacks
│   └── [name]_stack.tiff
├── 2JPG/                         # Intermediate: JPG sequences
│   └── [name]_stack/
│       ├── 000000.jpg
│       └── ...
├── 3CROP/                        # Output: Cropped videos
│   └── [name]_stack_crop.tif
├── 4CROP_JPG/                    # Cropped as JPG (if needed)
│   └── [name]_crop/
├── 5RIA_SEGMENT/                 # Segmentation outputs
│   ├── [name].h5                 # Mask data
│   └── [name]_overlay.mp4        # QC video
├── angles/                       # Angle analysis
│   ├── [name]_angle.csv          # Raw angles
│   ├── [name]_angle_smooth.csv   # Smoothed
│   ├── [name]_polar.csv          # Polar coordinates
│   └── [name]_skeleton.avi       # Annotated video
└── BAR269/                       # Final processed data
    ├── 1_final.csv
    └── ...
```

---

## Important Notes

### Hardware Requirements
- **GPU strongly recommended** for SAM2 (steps 3-4)
  - CUDA-enabled GPU: ~10x faster
  - Apple Silicon (MPS): ~5x faster
  - CPU-only: Very slow for SAM2 steps
- **RAM:** 16 GB minimum, 32 GB recommended for large videos
- **Storage:** ~2-3 GB per video (all intermediate files)

### SAM2 Model Selection
**Step 3 (AutoCrop):** `sam2.1_hiera_tiny.pt`
- Fastest model
- Sufficient for initial cropping
- Lower memory usage

**Step 4 (MaskGen):** `sam2.1_hiera_base_plus.pt`
- More accurate segmentation
- Better for small object tracking
- Required for precise RIA boundary detection

### Parameter Tuning

**For noisy videos:**
- Increase Gaussian blur kernel in step 5 (line 176)
- Increase `WINDOW_SIZE` in step 6 (more smoothing)

**For fast movements:**
- Decrease `CHUNK_SIZE` in steps 3-4 (more frequent re-prompting)
- Decrease `QUALITY_CHECK_INTERVAL` (more frequent QC)

**For poor segmentation:**
- Adjust morphological operations in step 5 (lines 180-182)
- Change `HEAD_POSITION` method
- Manually review and correct SAM2 prompts

### Quality Control Checkpoints

**After Step 2:**
- Check JPG quality in sample frames
- Verify normalization didn't lose detail

**After Step 3:**
- View cropped videos
- Ensure RIA region is centered
- Check for tracking losses

**After Step 4:**
- Watch overlay videos
- Verify nrD and nrV are correctly labeled
- Check mask statistics output

**After Step 5:**
- View skeleton video
- Verify head detection is consistent
- Check angle traces for jumps or artifacts

**After Step 6:**
- Compare smoothed vs. raw angles
- Ensure smoothing didn't over-blur rapid events

**After Step 7:**
- Check polar plots for expected structure
- Verify binning preserved behavioral features

---

## Advanced Usage

### Extracting Intensity from Masks

```python
import h5py
import tifffile
import numpy as np

# Load masks
with h5py.File('5RIA_SEGMENT/video.h5', 'r') as f:
    nrD_masks = f['masks/1'][:]
    nrV_masks = f['masks/2'][:]

# Load cropped video
video = tifffile.imread('3CROP/video_crop.tif')

# Extract intensities
nrD_intensity = []
nrV_intensity = []
for frame_idx in range(len(video)):
    frame = video[frame_idx]
    nrD_mask = nrD_masks[frame_idx, 0, :, :]
    nrV_mask = nrV_masks[frame_idx, 0, :, :]
    
    # Mean intensity within mask
    nrD_intensity.append(frame[nrD_mask].mean())
    nrV_intensity.append(frame[nrV_mask].mean())
```

### Correlating Neural Activity with Behavior

```python
import pandas as pd
import numpy as np

# Load data
intensity = pd.read_csv('intensity_data.csv')  # Your intensity extraction
polar = pd.read_csv('angles/video_polar.csv')

# Merge on frame
data = intensity.merge(polar, on='frame')

# Analyze activity by behavioral bin
binned_activity = data.groupby('bin_index')['nrD_intensity'].agg(['mean', 'std'])
```

### Batch Processing Multiple Videos

```python
import os
import glob

# Get all unprocessed videos
tiff_dir = '1TIFF'
processed = set([f.replace('_angle.csv', '') for f in os.listdir('angles') if f.endswith('_angle.csv')])
all_videos = set([f.replace('_stack.tiff', '') for f in os.listdir(tiff_dir) if f.endswith('_stack.tiff')])
to_process = all_videos - processed

# Process each
for video_name in sorted(to_process):
    print(f"Processing {video_name}...")
    # Update configuration in each script
    # Run pipeline...
```

---

## Troubleshooting

### SAM2 Installation Issues
- Follow official SAM2 installation guide
- Ensure PyTorch version matches SAM2 requirements
- For CUDA issues, verify CUDA toolkit version

### Memory Errors (Steps 3-4)
- Reduce `CHUNK_SIZE` parameter
- Process fewer videos simultaneously
- Close other applications
- Use smaller SAM2 model variant

### Segmentation Failures (Step 5)
- Check video contrast and brightness
- Adjust blur and threshold parameters
- Try different `HEAD_POSITION` options
- Manually review problematic frames

### Tracking Jumps or Losses (Steps 3-4)
- Provide additional manual prompts during processing
- Reduce `QUALITY_CHECK_INTERVAL` for more frequent checking
- Check for sudden illumination changes

### Angle Calculation Artifacts (Steps 5-7)
- Increase smoothing window size
- Check for head detection flips
- Verify midline ordering is consistent
- Filter out frames with poor segmentation

---

## Citation

If using this pipeline, please cite:
- SAM 2: Segment Anything in Images and Videos. https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

If using this pipeline for muscle analysis, please cite:
- A novel epifluorescence microscope design and software package to record naturalistic behaviour and cell activity in freely moving Caenorhabditis elegans
Sebastian N. Wittekindt, Hannah Owens, Lennard Wittekindt, Aurélie Guisnet, Michael Hendricks
bioRxiv 2025.03.21.644605; doi: https://doi.org/10.1101/2025.03.21.644605

---

## Contact

For questions about this pipeline, contact sebastian.wittekindt@gmail.com.
