import os
import tifffile
import numpy as np
import re

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def tiff_stacker(dir_path, output_path):
    # Get a list of all TIFF files in the directory
    tiff_files = [file for file in os.listdir(dir_path) if file.endswith('.tiff')]

    # Sort the TIFF files numerically
    tiff_files.sort(key=numerical_sort)

    # Read the first image to determine its shape and data type
    first_image = tifffile.imread(os.path.join(dir_path, tiff_files[0]))

    # Initialize an empty stack with appropriate dimensions and data type
    stack = np.zeros((len(tiff_files), first_image.shape[0], first_image.shape[1]), dtype=first_image.dtype)

    # Read each TIFF file and add it to the stack
    for i, tiff_file in enumerate(tiff_files):
        image = tifffile.imread(os.path.join(dir_path, tiff_file))
        stack[i] = image

    # Save the stack as a multi-image TIFF file
    tifffile.imwrite(output_path, stack)

    print(f"Multi-image TIFF stack saved at {output_path}")

# Specify the directory containing your TIF files
directory = r'4/4_reg'
name = 'GcaMP_stack'

if __name__ == "__main__":
    if os.path.isdir(directory):  # Ensure input_path is a directory
        output_path = os.path.join(directory, name + ".tiff") #output name
        parent_directory = os.path.dirname(directory)
        output_path = os.path.join(parent_directory, name + ".tiff") #output name
        tiff_stacker(directory, output_path)
    else:
        print(f"{directory} is not a directory")