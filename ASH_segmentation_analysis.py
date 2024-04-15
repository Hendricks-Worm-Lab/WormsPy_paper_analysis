import cv2
import numpy as np
import tifffile
import os
import csv
import imageio

def find_largest_area_and_brightest_pixels(raw,eightbit, threshold):
    # Apply threshold
    _, thresh = cv2.threshold(eightbit, threshold, 255, cv2.THRESH_BINARY) # TRY TRIANGLE

    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask of the largest contour
    mask = np.zeros_like(eightbit)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Find the area of the largest contour
    largest_area = cv2.contourArea(largest_contour)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(raw, raw, mask=mask)

    # Flatten the image and sort the pixel values in descending order
    sorted_pixels = np.sort(masked_image.flatten())[::-1]

    # Sum the brightest 50 pixels
    sum_brightest_50 = np.sum(sorted_pixels[:50])

    # mean of all pixels above 3000
    mean = np.mean(sorted_pixels[sorted_pixels > 3000])

    return largest_area, mask, sum_brightest_50, mean

# Specify the input file path
input_path = r"C:\Users\sebas\Documents\Segmentation\ASH_tiff_stacks\worm8_goodstack.tiff"
name = "worm8"

# Specify the output file path
output_folder = r"C:\Users\sebas\Documents\Segmentation\ASH_analyzed"
# Read the image file
stack = tifffile.imread(input_path)

# Ensure the image data is 16-bit
stack = stack.astype(np.uint16)

output_filename = name + '_thresh.gif'

# Check if the output file already exists and delete it
if os.path.exists(output_filename):
    os.remove(output_filename)

frames = []  # List to store frames for the GIF

prev_start_point = None
csv_path = os.path.join(output_folder, name + ".csv")
if os.path.exists(csv_path):
    os.remove(csv_path)

with open(csv_path, "a", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["frame", "largest area", "sum of brightest 50 pixels", "mean of all pixels"])

    frame_number = 0

    # Loop through each image in the stack
    for i, frame in enumerate(stack):
        # Convert to gray, median filter, threshold
        raw = frame
        # convert frame to 8bit
        eightbit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        blurred = cv2.bilateralFilter(eightbit, 15, 150, 150)

        #call the function to find the largest area and brightest pixels
        largest_area, mask, sum_brightest_50, mean = find_largest_area_and_brightest_pixels(raw,blurred, 40)

        csv_writer.writerow([frame_number, largest_area, sum_brightest_50, mean])
        frame_number += 1

        # Add the mask to eightbit
        eightbit = cv2.addWeighted(eightbit, 1, mask, 1, 0)

        # Display the output frame by frame
        cv2.imshow('output', eightbit)
        if cv2.waitKey(30) == ord('q'):
            break
        
        rgb_frame = cv2.cvtColor(eightbit, cv2.COLOR_GRAY2RGB)
        frames.append(rgb_frame)

    # Save frames as a GIF
    output_path = os.path.join(output_folder, output_filename)
    imageio.mimsave(output_path, frames, 'GIF', fps=10)