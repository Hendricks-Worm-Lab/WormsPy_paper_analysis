import os
import tifffile
import numpy as np

# Specify the directory containing your TIF files
dir_path = 'C:/Users/sebastian/Downloads/worm1'

# Get a list of all TIFF files in the directory
tiff_files = [file for file in os.listdir(dir_path) if file.endswith('.tif')]

# Read the first image to determine its shape
first_image = tifffile.imread(os.path.join(dir_path, tiff_files[0]))

# Initialize an empty stack with the appropriate dimensions
stack = np.zeros((len(tiff_files), first_image.shape[0], first_image.shape[1]), dtype=np.uint16)

# Read each TIFF file and add it to the stack
for i, tiff_file in enumerate(tiff_files):
    stack[i, :, :] = tifffile.imread(os.path.join(dir_path, tiff_file))

# Save the stack as a multi-image TIFF file
output_path = 'C:/Users/sebastian/Downloads'
tifffile.imsave(output_path, stack)

print(f"Multi-image TIFF stack saved at {output_path}")