# Muscle Calcium Imaging Analysis Pipeline

This folder contains a script for analyzing muscle calcium activity along the body wall of *C. elegans* using midline-based segmentation and orthogonal intensity profiling.

## Overview

The pipeline analyzes brightfield videos of worms expressing muscle calcium indicators (e.g., myo-3::GCaMP). It automatically detects the worm body, computes the midline skeleton, and extracts calcium signals from orthogonal segments along the anterior-posterior axis. This allows for spatiotemporal mapping of muscle activity patterns.

## Required Packages

- Python 3.x
- opencv-cv2 (cv2)
- numpy
- scikit-image (skimage.morphology)
- os
- csv

## Pipeline Overview

**Single Script:** `myo3gcamp.py`

This script performs all processing steps:
1. Worm body segmentation from brightfield
2. Midline skeletonization
3. Generation of evenly-spaced points along midline
4. Calculation of orthogonal intensity profiles
5. Export of spatiotemporal data to CSV

---

## Script Documentation

### myo3gcamp.py

**Purpose:** Extracts spatiotemporal muscle calcium signals from brightfield videos by measuring fluorescence intensity in orthogonal segments along the worm's midline.

**Input:**
- Brightfield AVI video (must specify path in script)

**Output:**
- `midline_points.csv` - Detailed spatiotemporal data
- `output.avi` - Annotated video showing segmentation and measurement regions

---

### Configuration Parameters

**Critical - Must Configure:**
```python
# Line 88: Input video path
cap = cv2.VideoCapture(r"PATH/TO/YOUR/VIDEO.avi")
```

**Segmentation Parameters (user-adjustable):**
```python
midline_segments = 20    # Number of points along the midline (line 7)
ortho_length = 40        # Length of orthogonal sampling segments in pixels (line 8)
```

**Recommended Settings:**
- `midline_segments = 20`: Provides good spatial resolution along body
- `ortho_length = 40`: Captures full muscle width in most orientations

---

### Algorithm Details

#### 1. Worm Segmentation
**Method:** Adaptive thresholding on median-filtered grayscale
```python
gray = cv2.medianBlur(gray_raw, 21)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
```

**Process:**
- Median blur (21×21 kernel) removes noise
- Triangle thresholding automatically finds optimal threshold
- Largest contour selected as worm body

#### 2. Midline Extraction
**Method:** Morphological skeletonization
```python
skeleton = skeletonize(worm_binary.squeeze()).astype(np.uint8)
```

**Process:**
- Binary mask of worm body
- Skeletonize to 1-pixel-wide midline
- Convert to contour for ordered point extraction

#### 3. Midline Ordering
**Function:** `contour_to_polyline()`

**Process:**
- Finds the two endpoints (maximum pairwise distance)
- Orders points from one end to the other using nearest-neighbor
- Maintains consistent head-tail direction across frames

#### 4. Even Sampling
**Function:** `generate_evenly_spaced_points()`

**Process:**
- Calculates cumulative distances along midline
- Interpolates `midline_segments` evenly-spaced points
- Ensures consistent spatial sampling regardless of worm pose

#### 5. Angle Calculation
**Function:** `three_point_angle()`

**Process:**
- For each point, calculates angle between neighboring points
- Provides local body curvature information
- Used for behavioral analysis and quality control

#### 6. Orthogonal Profiling
**Function:** `find_orthogonal_endpoints()`

**Process:**
- For each midline point, calculates perpendicular line
- Extracts pixel intensities along `ortho_length`
- Samples intensity from body left to body right

---

### Output Data Structure

#### CSV File: `midline_points.csv`

**Columns:**
- `frame`: Frame number (0-indexed)
- `order`: Position along body (1 to `midline_segments-2`)
- `x`, `y`: Pixel coordinates of midline point
- `angle`: Local body angle in degrees
- `segment_position`: Normalized position along orthogonal segment (0-1)
- `intensity`: Grayscale intensity value

**Data Organization:**
- Each frame contains `(midline_segments - 2) × ortho_length` rows
- `order` indicates anterior-posterior position
- `segment_position` indicates left-right position across body

**Example rows:**
```
frame,order,x,y,angle,segment_position,intensity
0,1,245,312,165.2,0.000,45
0,1,245,312,165.2,0.025,52
0,1,245,312,165.2,0.050,58
...
0,2,248,315,168.1,0.000,41
...
```

#### Video File: `output.avi`

**Visual Elements:**
- Green outline: Worm body contour
- Magenta line: Midline skeleton
- Yellow circles: Sampled midline points (numbered)
- Red lines: Orthogonal intensity sampling segments

**Properties:**
- Codec: MJPG
- Frame rate: 10 FPS
- Resolution: Same as input video
- Quality: 100 (maximum)

---

## Usage Instructions

### Step-by-Step Workflow

1. **Prepare Video:**
   - Ensure video is in AVI format
   - Brightfield channel should show worm body clearly
   - Background should be relatively uniform

2. **Configure Script:**
   ```python
   # Edit line 88:
   cap = cv2.VideoCapture(r"path/to/your/video.avi")
   
   # Optionally adjust sampling:
   midline_segments = 20    # More = finer spatial resolution
   ortho_length = 40        # Should cover full body width
   ```

3. **Run Script:**
   ```bash
   python myo3gcamp.py
   ```

4. **Monitor Execution:**
   - Window displays real-time processing
   - Press 'q' to quit early (saves progress)
   - Processing time: ~1-2 seconds per frame

5. **Check Outputs:**
   - Review `output.avi` for segmentation quality
   - Verify `midline_points.csv` has expected number of rows

---

## Data Analysis

### Loading and Reshaping Data

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('midline_points.csv')

# Get dimensions
n_frames = df['frame'].max() + 1
n_orders = df['order'].nunique()
n_positions = df['segment_position'].nunique()

# Reshape to 3D array: [frames, body_position, width]
data = df.pivot_table(
    index=['frame', 'order'],
    columns='segment_position',
    values='intensity'
)
```

### Common Analyses

**1. Spatiotemporal Heatmap:**
```python
import matplotlib.pyplot as plt

# Average across width for each body position and frame
heatmap_data = df.groupby(['frame', 'order'])['intensity'].mean().unstack()

plt.figure(figsize=(12, 6))
plt.imshow(heatmap_data.T, aspect='auto', cmap='hot', origin='lower')
plt.xlabel('Frame')
plt.ylabel('Body Position (Anterior → Posterior)')
plt.title('Muscle Activity Along Body')
plt.colorbar(label='Intensity')
plt.show()
```

**2. Mean Activity Over Time:**
```python
# Average all spatial positions per frame
mean_activity = df.groupby('frame')['intensity'].mean()

plt.plot(mean_activity)
plt.xlabel('Frame')
plt.ylabel('Mean Intensity')
plt.title('Whole-Body Muscle Activity')
plt.show()
```

**3. Anterior vs. Posterior Activity:**
```python
# Split body into regions
df['region'] = df['order'].apply(
    lambda x: 'anterior' if x < 10 else 'posterior'
)

# Compare regions
region_activity = df.groupby(['frame', 'region'])['intensity'].mean().unstack()

plt.plot(region_activity['anterior'], label='Anterior')
plt.plot(region_activity['posterior'], label='Posterior')
plt.xlabel('Frame')
plt.ylabel('Mean Intensity')
plt.legend()
plt.show()
```

---

## Data Organization

### Input/Output Structure
```
Muscle/
├── myo3gcamp.py              # Main analysis script
├── HBR4/                     # Strain/condition 1
│   ├── worm1.csv
│   ├── worm2.csv
│   └── ...
└── MMH116/                   # Strain/condition 2
    ├── worm1.csv
    ├── worm2.csv
    └── ...
```

**Note:** CSV files in strain folders are final processed outputs. Generate them by running `myo3gcamp.py` on each video and manually organizing results.

---

## Important Notes

### Video Requirements
- **Format:** AVI (other formats may work but untested)
- **Content:** Brightfield channel showing worm body
- **Quality:** Clear worm outline, minimal debris
- **Background:** Relatively uniform (not strictly required due to triangle thresholding)

### Parameter Tuning

**When to adjust `midline_segments`:**
- Increase (e.g., 30) for longer worms or finer spatial resolution
- Decrease (e.g., 15) for shorter worms or faster processing
- Must be ≥ 3 (code uses points 1 to n-2)

**When to adjust `ortho_length`:**
- Should span full body width at typical orientations
- If segments miss body edges: increase value
- If segments extend far beyond body: decrease value
- Typical range: 30-50 pixels for standard microscopy

**When segmentation fails:**
- Worm too dim: Adjust video exposure or preprocessing
- Complex background: Pre-process video (background subtraction)
- Multiple worms: Manually crop video or track single worm

### Processing Considerations

**Memory:**
- Large videos may require significant RAM
- CSV file size: ~1-2 KB per frame
- Consider processing in batches for very long recordings

**Speed:**
- ~1-2 seconds per frame (depends on hardware)
- Skeletonization is the slowest step
- Close the display window (`cv2.imshow`) for faster processing

**Quality Control:**
- Always review `output.avi` before analyzing data
- Check for:
  - Consistent worm detection across frames
  - Smooth midline tracking (no jumps)
  - Orthogonal segments spanning body width
  - Consistent head-tail orientation

---

## Troubleshooting

### Segmentation Issues

**Problem:** Worm not detected
- **Solution:** Check video quality; ensure sufficient contrast
- Try adjusting median blur kernel size (line 107)

**Problem:** Background noise detected as worm
- **Solution:** Add area filtering to contour detection
- Example: `if cv2.contourArea(worm_contour) < 1000: continue`

**Problem:** Multiple contours detected
- **Solution:** Code takes largest contour by default
- Ensure worm is largest object in frame

### Midline Issues

**Problem:** Midline jumps or disconnected
- **Solution:** Issue with skeletonization; check worm segmentation first
- May need to fill holes in binary mask before skeletonization

**Problem:** Inconsistent head-tail orientation
- **Solution:** Adjust `prev_start_point` tracking logic (lines 140-146)
- May need to implement more robust orientation tracking

### Orthogonal Profiling Issues

**Problem:** Division by zero error in `find_orthogonal_endpoints()`
- **Solution:** Occurs when midline is perfectly horizontal
- Add small epsilon to slope calculation

**Problem:** Intensities are zero
- **Solution:** Orthogonal segments outside image bounds
- Check `ortho_length` parameter

### Output Issues

**Problem:** CSV file is empty or incomplete
- **Solution:** Script exited early; check for errors in console
- Ensure video path is correct and video is readable

**Problem:** Video file won't open
- **Solution:** Check codec compatibility
- Try different video player or convert output format

---

## Advanced Modifications

### Custom Intensity Metrics

To calculate different intensity statistics:

```python
# After line 161, modify to calculate mean, max, or other metrics:
circle_pixels = gray_raw[circle_mask]
mean_intensity = np.mean(circle_pixels)
max_intensity = np.max(circle_pixels)
# Add to csv_writer.writerow()
```

### Export Spatial Maps

To save 2D intensity maps per frame:

```python
# After extracting intensity profile:
intensity_map = np.zeros((midline_segments, ortho_length))
for order_idx, intensities in enumerate(profiles):
    intensity_map[order_idx, :] = intensities

# Save as image or numpy array
np.save(f'frame_{frame_number}_map.npy', intensity_map)
```

---

## Citation

If using this pipeline for muscle analysis, please cite:
- A novel epifluorescence microscope design and software package to record naturalistic behaviour and cell activity in freely moving Caenorhabditis elegans
Sebastian N. Wittekindt, Hannah Owens, Lennard Wittekindt, Aurélie Guisnet, Michael Hendricks
bioRxiv 2025.03.21.644605; doi: https://doi.org/10.1101/2025.03.21.644605

---

## Contact

For questions about this pipeline, contact sebastian.wittekindt@gmail.com.
