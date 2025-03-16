import cv2
import numpy as np
import glob
import os
import tifffile
import matplotlib.pyplot as plt
import pandas as pd

def apply_rotation_from_csv(img, angle, visualize):
    """
    Apply rotation to an image based on angle from CSV.
    
    Args:
        img: Image to rotate
        angle: Rotation angle in degrees
        center: Center of rotation (optional)
        visualize: Whether to show visualization
        
    Returns:
        rotated_img: Rotated image
    """
    # Store original shape
    h, w = img.shape[:2]
    
    center = (w // 2, h // 2)

    # Apply negative angle to rotate the image back to atan2 = 0
    # This counter-rotates the image to align all images to the same orientation
    rotation_angle = -angle
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Apply rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    
    # Visualize the rotation process if requested
    if visualize:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        
        plt.subplot(132)
        plt.imshow(rotated_img, cmap='gray')
        plt.title(f'Rotated Image (angle={angle:.2f}°)')
        
        # Show difference image
        plt.subplot(133)
        diff = cv2.absdiff(img, rotated_img)
        plt.imshow(diff, cmap='hot')
        plt.title('Difference')
        
        plt.tight_layout()
        # plt.draw()
        plt.pause(0.1)  # Pause for 50ms

    return rotated_img

def main():
    # Define the folder containing the .tif images
    folder = "RIA_sinusoidal/10"
    name = 10
    image_paths = sorted(glob.glob(os.path.join(folder, "*.tiff")))
    
    if not image_paths:
        print("No .tiff images found in the folder.")
        return
    
    # Load CSV file with angle measurements
    try:
        csv_path = 'angles/' + str(name) + "_angle_terp.csv"  # Path to your CSV file
        angles_df = pd.read_csv(csv_path)
        print(f"Loaded angles data for {len(angles_df)} frames")
        
        # Check if CSV contains required columns
        if "frame" not in angles_df.columns or "RIA_angle" not in angles_df.columns:
            print("CSV must contain 'frame' and 'angle' columns")
            return
            
        # Group by frame and calculate mean angle if multiple entries per frame
        angles_by_frame = angles_df.groupby("frame")["RIA_angle"].mean().to_dict()
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Create output folder
    parent_folder = os.path.dirname(folder)
    rotated_folder = os.path.join(parent_folder, str(name) + "_aligned")
    os.makedirs(rotated_folder, exist_ok=True)
    
    # Statistics
    rotation_stats = {
        'total': len(image_paths),
        'rotated': 0,
        'skipped': 0
    }
    
    # Process each image
    for i, path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {path}")
        
        # Check if we have angle data for this frame
        if i not in angles_by_frame:
            print(f"No RIA_angle data for frame {i}, skipping rotation")
            img_file = tifffile.imread(path)
            rotated = img_file  # Use original image
            rotation_stats['skipped'] += 1
            status = "SKIPPED (No RIA_angle data)"
        else:
            # Load the image
            img_file = tifffile.imread(path)
            img_file = img_file.astype(np.uint16)
            img = np.array(img_file)
            
            # Get angle for this frame and apply rotation
            angle = angles_by_frame[i]
            
            # Set visualize=True to see the rotation process for debugging
            visualize = (i % 20 == 0)  # Visualize only every 20th frame
            
            try:
                rotated = apply_rotation_from_csv(img, angle, visualize=visualize)
                rotation_stats['rotated'] += 1
                status = f"ROTATED (angle: {angle:.2f}°)"
                
            except Exception as e:
                print(f"Rotation failed: {e}")
                rotated = img  # Use original image if rotation fails
                rotation_stats['skipped'] += 1
                status = "FAILED"
        
        # Scale the rotated image back to 16-bit if needed
        rotated_uint16 = np.clip(rotated, 0, 65535).astype(np.uint16)
        
        # Save the rotated image
        out_path = os.path.join(rotated_folder, f"{i:03d}_" + os.path.basename(path))
        if os.path.exists(out_path):
            os.remove(out_path)
        tifffile.imwrite(out_path, rotated_uint16)
        
        print(f"Status: {status}")
        print(f"Image saved to: {out_path}")
    
    # Print statistics
    print("\n------ Rotation Statistics ------")
    print(f"Total images: {rotation_stats['total']}")
    print(f"Rotated: {rotation_stats['rotated']} ({rotation_stats['rotated']/rotation_stats['total']*100:.1f}%)")
    print(f"Skipped/Failed: {rotation_stats['skipped']} ({rotation_stats['skipped']/rotation_stats['total']*100:.1f}%)")
    print("--------------------------------")

if __name__ == "__main__":
    main()