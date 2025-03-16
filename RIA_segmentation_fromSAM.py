import os
import pickle
import numpy as np
import pandas as pd
import tifffile
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def create_circular_mask(image_shape, center, radius):
    """Create a circular mask with specified center and radius"""
    y, x = np.ogrid[:image_shape[0], :image_shape[1]]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = dist_from_center <= radius
    return mask

def process_frames_with_masks(tif_path, mask_path):
    """
    Process a TIF stack with masks from a PKL file and calculate statistics
    """
    # Load the TIF stack
    print(f"Loading TIF stack from {tif_path}...")
    tif_stack = tifffile.imread(tif_path)
    num_frames = tif_stack.shape[0]
    print(f"Loaded {num_frames} frames with shape {tif_stack.shape[1:]}.")
    
    # Load the masks
    print(f"Loading masks from {mask_path}...")
    with open(mask_path, 'rb') as f:
        masks_by_frame = pickle.load(f)
    print(f"Loaded masks for {len(masks_by_frame)} frames.")
    
    # Debug: Check structure of first frame masks
    first_frame = sorted(masks_by_frame.keys())[0]
    print(f"\nExamining masks in first frame ({first_frame}):")
    for obj_id in masks_by_frame[first_frame]:
        mask = masks_by_frame[first_frame][obj_id]
        mask_array = np.array(mask)
        true_count = np.sum(mask_array)
        total_size = mask_array.size
        percent = (true_count / total_size) * 100
        print(f"  Object {obj_id}: Shape={mask_array.shape}, dtype={mask_array.dtype}")
        print(f"  True values: {true_count}/{total_size} ({percent:.2f}%)")

    # Initialize results dictionary with new metrics
    results = {
        'frame': [],
        'nrD_top25_mean': [],
        'nrD_top5_mean': [],
        'nrV_top25_mean': [],
        'nrV_top5_mean': []
    }
    
    # Set up figure for visualization
    plt.figure(figsize=(15, 5))
    plt.ion()  # Turn on interactive mode
    
    # Process each frame
    for frame_num in sorted(masks_by_frame.keys()):
        if frame_num >= num_frames:
            print(f"Warning: Mask for frame {frame_num} exists but frame doesn't exist in TIF stack.")
            continue
        
        frame = tif_stack[frame_num]
        masks = masks_by_frame[frame_num]
        
        results['frame'].append(frame_num)
        
        # Create visualization with subplots for better debugging
        plt.clf()  # Clear previous frame
        
        # Create 3 subplots: original frame, nrD mask, nrV mask
        plt.subplot(1, 3, 1)
        plt.imshow(frame, cmap='gray')
        plt.title(f"Frame {frame_num}")
        
        # Process each object ID (2 = nrD, 3 = nrV)
        for idx, (obj_id, column_prefix) in enumerate({2: 'nrD', 3: 'nrV'}.items()):
            plt.subplot(1, 3, idx+2)
            plt.imshow(frame, cmap='gray')
            plt.title(f"{column_prefix} (ID: {obj_id})")
            
            if obj_id in masks:
                # Apply mask to frame
                mask = masks[obj_id]
                # Check mask dimensions and reshape if needed
                if len(np.array(mask).shape) > 2:
                    # Convert to numpy array and reshape to 2D
                    mask = np.array(mask).reshape(frame.shape)
                # Ensure mask is boolean type
                if np.array(mask).dtype != bool:
                    mask = np.array(mask).astype(bool)
                
                # Display original mask outline
                plt.contour(mask, colors='blue', linewidths=0.5, alpha=0.7)
                
                # Calculate midpoint of the mask
                if np.sum(mask) > 0:  # Make sure mask has some true values
                    # Get coordinates of True values
                    rows, cols = np.where(mask)
                    # Calculate midpoint
                    mid_y = int(np.mean(rows))
                    mid_x = int(np.mean(cols))
                    
                    # Create circular mask with radius 5
                    circle_mask = create_circular_mask(frame.shape, (mid_y, mid_x), 5)
                    
                    # Draw the circle on the plot
                    circle = Circle((mid_x, mid_y), 5, fill=False, edgecolor='red', linewidth=2)
                    plt.gca().add_patch(circle)
                    plt.plot(mid_x, mid_y, 'rx')  # Mark the midpoint
                    
                    # Calculate statistics for pixels within the circular mask
                    circle_pixels = frame[circle_mask]
                    
                    # Calculate new metrics: top 25 and top 5 pixel means
                    if len(circle_pixels) >= 25:
                        # Get top 25 brightest pixels
                        top_25 = np.sort(circle_pixels)[-25:]
                        top_25_mean = np.mean(top_25)
                        
                        # Get top 5 brightest pixels
                        top_5 = np.sort(circle_pixels)[-5:]
                        top_5_mean = np.mean(top_5)
                    elif len(circle_pixels) >= 5:
                        # If fewer than 25 but at least 5 pixels
                        top_25_mean = np.mean(circle_pixels)
                        
                        # Get top 5 brightest pixels
                        top_5 = np.sort(circle_pixels)[-5:]
                        top_5_mean = np.mean(top_5)
                    else:
                        # If fewer than 5 pixels, use all available
                        top_25_mean = np.mean(circle_pixels) if len(circle_pixels) > 0 else np.nan
                        top_5_mean = np.mean(circle_pixels) if len(circle_pixels) > 0 else np.nan
                    
                    plt.title(f"{column_prefix}: Circle r=5 at ({mid_x}, {mid_y})")
                    
                else:
                    # No true values in mask
                    top_25_mean = np.nan
                    top_5_mean = np.nan
                    plt.title(f"{column_prefix}: Empty mask")
                
                # Store the new metrics
                results[f'{column_prefix}_top25_mean'].append(top_25_mean)
                results[f'{column_prefix}_top5_mean'].append(top_5_mean)
            else:
                # If no mask for this object ID in this frame
                plt.title(f"{column_prefix}: No mask")
                results[f'{column_prefix}_top25_mean'].append(np.nan)
                results[f'{column_prefix}_top5_mean'].append(np.nan)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)  # Pause for 50ms
    
    plt.ioff()  # Turn off interactive mode
    plt.close()  # Close the figure
    
    # Create DataFrame and return
    return pd.DataFrame(results)

def main():
    # Get the directory containing both TIF and PKL files
    data_dir = input("Enter the directory containing .tif and .pkl files: ").strip()
    
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a valid directory.")
        return
    
    # Find TIF files
    tif_files = glob(os.path.join(data_dir, "*.tif"))
    tif_files.extend(glob(os.path.join(data_dir, "*.tiff")))
    
    # Find PKL files
    pkl_files = glob(os.path.join(data_dir, "*.pkl"))
    
    if not tif_files:
        print("No TIF files found in the directory.")
        return
    
    if not pkl_files:
        print("No PKL files found in the directory.")
        return
    
    # Display found files
    print("Found TIF files:")
    for i, file_path in enumerate(tif_files):
        print(f"{i+1}. {os.path.basename(file_path)}")
    
    print("\nFound PKL files:")
    for i, file_path in enumerate(pkl_files):
        print(f"{i+1}. {os.path.basename(file_path)}")
    
    # Get user selection
    tif_idx = int(input("\nSelect TIF file number: ")) - 1
    pkl_idx = int(input("Select PKL file number: ")) - 1
    
    if not (0 <= tif_idx < len(tif_files) and 0 <= pkl_idx < len(pkl_files)):
        print("Invalid selection.")
        return
    
    tif_path = tif_files[tif_idx]
    pkl_path = pkl_files[pkl_idx]
    
    # Process the files
    results_df = process_frames_with_masks(tif_path, pkl_path)
    
    # Save results to CSV
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    csv_path = os.path.join(data_dir, f"{base_name}_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Display a preview of the results
    print("\nResults preview:")
    print(results_df.head())

if __name__ == "__main__":
    main()