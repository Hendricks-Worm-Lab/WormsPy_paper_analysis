import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import filters
import os
import csv
from tifffile import imread

input_filename = "default.tiff"

midline_segments = 20
ortho_length = 40

########## FUNCTIONS #############

#this ensures that the points along the midline are ordered and returns them as connected segments
def order_points_by_nearest_neighbor(points, start_point=None):
    if start_point is None:
        start_point = points[0]
    else:
        start_point = tuple(start_point)

    polyline = [start_point]
    remaining_points = set(map(tuple, points))
    remaining_points.remove(start_point)
    while remaining_points:
        last_point = polyline[-1]
        next_point = min(remaining_points, key=lambda p: np.linalg.norm(np.array(p) - last_point))
        polyline.append(next_point)
        remaining_points.remove(next_point)
    return np.array(polyline)

def generate_evenly_spaced_points(polyline, num_points):
    distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    target_distances = np.linspace(0, distances[-1], num_points)
    interpolated_points = np.zeros((num_points, 2), dtype=int)
    for i, target_distance in enumerate(target_distances):
        idx = np.searchsorted(distances, target_distance, side='right')
        if idx == 0:
            interpolated_points[i] = polyline[0]
        elif idx == len(polyline):
            interpolated_points[i] = polyline[-1]
        else:
            t = (target_distance - distances[idx - 1]) / (distances[idx] - distances[idx - 1])
            interpolated_points[i] = polyline[idx - 1] + t * (polyline[idx] - polyline[idx - 1])
    return interpolated_points

def three_point_angle(a, b, c):
    ba = np.array(a) - np.array(b)  # vector from b to a
    bc = np.array(c) - np.array(b)  # vector from b to c

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle) 
                              
#these are the endpoints of the little line segments across the midline through each point
def find_orthogonal_endpoints(a, b, p, L):
    # Create a vector AB from point A to point B
    AB = np.array(b) - np.array(a)
    
    # Normalize AB to get the unit vector u_AB
    u_AB = AB / np.linalg.norm(AB)
    
    # Rotate u_AB by 90 degrees to get the orthogonal unit vector u_orthogonal
    u_orthogonal = np.array([-u_AB[1], u_AB[0]])
    
    # Multiply u_orthogonal by L/2 to get half_length_vector
    half_length_vector = (L / 2) * u_orthogonal
    
    # Add and subtract half_length_vector from p to get the endpoints
    point = np.array(p)
    endpoint_1 = point + half_length_vector
    endpoint_2 = point - half_length_vector
    
    return endpoint_1, endpoint_2


############ MAIN ##############

# load tiff
stack = imread(r"D:\Your\Directory\\" + input_filename)
output_filename = input_filename + '_output.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Check if the output file already exists and delete it
if os.path.exists(output_filename):
    os.remove(output_filename)

out = cv2.VideoWriter(output_filename, fourcc, 10.0, (stack.shape[2], stack.shape[1]))
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

with open(input_filename + "midline_points.csv", "a", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["frame", "order", "x", "y", "angle", "segment_position", "intensity"])

    frame_number = 0

    ##################### MAIN LOOP #####################
    while frame_number < stack.shape[0]:
        
        frame = stack[frame_number]
        blurred = filters.gaussian(frame, sigma=5.0, preserve_range=True)
        gray = cv2.convertScaleAbs(blurred, alpha=255/stack.max())
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        # manually review thresholding
        #cv2.imshow('thresholded', thresh)
        #cv2.waitKey(0)
        
        # create an 8-bit color version of frame to overlay segmentation output
        color_frame = cv2.convertScaleAbs(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), alpha=255/stack.max())
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the worm (biggest conotour) and draw it on the color frame
            worm_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(color_frame, [worm_contour], -1, (0, 255, 0), 1)
                        
            #Skeletonize the worm contour to find midline, create midline contour
            worm_binary = np.zeros_like(thresh)
            cv2.drawContours(worm_binary, [worm_contour], -1, 255, -1)
            skeleton = skeletonize(worm_binary).astype(np.uint8)

            # Find the coordinates of the non-zero pixels in the skeleton
            y, x = np.nonzero(skeleton)

            # Combine the x and y coordinates into a 2D array
            midline_polyline = np.column_stack((x, y))

            # Calculate the number of neighbors for each point
            neighbors = cv2.filter2D(skeleton, -1, np.ones((3, 3)))

            # Find the points with only one neighbor (the endpoints)
            endpoints = midline_polyline[neighbors[midline_polyline[:, 1], midline_polyline[:, 0]] == 2]

            # Ensure the nose has the lower y-coordinate
            nose_point, tail_point = sorted(endpoints, key=lambda point: point[1])

            # Draw the nose and tail points on color_frame
            cv2.circle(color_frame, tuple(nose_point), 5, (255, 255, 255), -1)  # white nose
            cv2.circle(color_frame, tuple(tail_point), 5, (255, 0, 0), -1)  # blue tail
            
            # Sort the points in midline_polyline by nearest neighbor
            midline_polyline = order_points_by_nearest_neighbor(midline_polyline, nose_point)

            # Draw midline_polyline on color_frame in blue
            cv2.polylines(color_frame, [midline_polyline], isClosed=False, color=(0, 255, 255), thickness=1)

            #evenly spaced points along the midline
            midline_points = generate_evenly_spaced_points(midline_polyline, midline_segments)
                        
            # Loop through midline points
            for order, point in enumerate(midline_points[1:-1], start=1):
                x, y = int(round(point[0])), int(round(point[1]))
                cv2.circle(color_frame, (x, y), 3, (0, 255, 255), -1)

                # Calculate angle formed by current point and neighboring points
                prev_point = midline_points[order - 1]
                next_point = midline_points[order + 1]
                point_angle = three_point_angle(point, next_point, prev_point)

                # Find orthogonal endpoints
                start, end = find_orthogonal_endpoints(prev_point, next_point, point, ortho_length)

                # Draw the orthogonal line segments
                if start[1] is not None and end[1] is not None:
                    
                    # Draw line on the output image
                    cv2.line(color_frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (0, 0, 255), 1)

                    # Extract pixel intensity profile along the orthogonal line segment
                    x_values = np.linspace(int(start[0]), int(end[0]), ortho_length)
                    y_values = np.linspace(int(start[1]), int(end[1]), ortho_length)
                    pixel_intensity_values = frame[y_values.astype(int), x_values.astype(int)]

                    # Write to csv
                    for position, intensity in zip(np.linspace(0, 1, len(pixel_intensity_values), endpoint=False), pixel_intensity_values):
                        csv_writer.writerow([frame_number, order, point[0], point[1], point_angle, position, intensity])


            frame_number += 1

            # Display the output frame by frame
            cv2.imshow('output', color_frame)
            if cv2.waitKey(30) == ord('q'):
                break

        out.write(color_frame)

    out.release()