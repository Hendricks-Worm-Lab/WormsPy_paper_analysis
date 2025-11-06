import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.morphology import thin
import os
import csv
import numbers

midline_segments = 20
ortho_length = 40

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

#takes the midline contour and makes it a polyline
def contour_to_polyline(contour, prev_start_point=None):
    contour = contour.squeeze()                #gets rid of danglers I think
    contour = np.unique(contour, axis=0)       #dedupe
    pairwise_distances = np.sum((contour[:, np.newaxis] - contour[np.newaxis, :]) ** 2, axis=2)
    np.fill_diagonal(pairwise_distances, -1)
    end1, end2 = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)

    if prev_start_point is None:
        start_point = contour[end1]
    else:
        start_point = contour[end1] if np.linalg.norm(prev_start_point - contour[end1]) < np.linalg.norm(prev_start_point - contour[end2]) else contour[end2]

    polyline = order_points_by_nearest_neighbor(contour, start_point)
    return polyline


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

def three_point_angle(a,b,c):
    next_i = (b[0] - a[0]) / np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    next_j = (b[1] - a[1]) / np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    prev_i = (c[0] - a[0]) / np.sqrt((c[0] - a[0])**2 + (c[1] - a[1])**2)
    prev_j = (c[1] - a[1]) / np.sqrt((c[0] - a[0])**2 + (c[1] - a[1])**2)
    angle = np.degrees(np.arccos(next_i * prev_i + next_j * prev_j))
    return angle      
                              
#these are the endpoints of the little line segments across the midline through each point
def find_orthogonal_endpoints(a, b, p, L):
    # Calculate the slope of line segment A
    slope_A = (b[1] - a[1]) / (b[0] - a[0])
    
    # Calculate the negative reciprocal of the slope
    slope_orthogonal = -1 / slope_A
    
    # Calculate the unit vector in the direction of the orthogonal line
    length = np.sqrt(1 + slope_orthogonal ** 2)
    unit_vector = np.array([1 / length, slope_orthogonal / length])
    
    # Multiply the unit vector by half of the desired length (L/2)
    half_length_vector = (L / 2) * unit_vector
    
    # Add and subtract the resulting vector from the point (x, y)
    point = np.array(p)
    endpoint_1 = point + half_length_vector
    endpoint_2 = point - half_length_vector
    
    return endpoint_1, endpoint_2

# Open the video file
cap = cv2.VideoCapture(r"")

# Read the first frame
ret, frame = cap.read()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_filename = 'output.avi'

# Check if the output file already exists and delete it
if os.path.exists(output_filename):
    os.remove(output_filename)

out = cv2.VideoWriter(output_filename, fourcc, 10.0, (frame.shape[1], frame.shape[0]))
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

prev_start_point = None

with open("midline_points.csv", "a", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["frame", "order", "x", "y", "angle", "segment_position", "intensity"])

    frame_number = 0

    #looping through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to gray, median filter, threshold
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray_raw, 21)  # 21 is 2 * 10 + 1
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        #Find largest contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            worm_contour = max(contours, key=cv2.contourArea)
            #Draw worm outline
            cv2.drawContours(frame, [worm_contour], -1, (0, 255, 0), 1)

            #Skeletonize the worm contour to find midline, create midline contour
            worm_binary = np.zeros_like(gray)
            cv2.drawContours(worm_binary, [worm_contour], -1, 255, -1)
            skeleton = skeletonize(worm_binary.squeeze()).astype(np.uint8)
            skeleton_binary = np.zeros_like(skeleton)
            skeleton_binary[skeleton == 1] = 255
            midline_contours, _ = cv2.findContours(skeleton_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(midline_contours) > 0:
                midline = max(midline_contours, key=lambda c: cv2.arcLength(c, False))
                #Draw the midline
                cv2.drawContours(frame, [midline], -1, (255, 0, 255), 1)
                #Convert the midline contour to a polyline
                midline_polyline = contour_to_polyline(midline, prev_start_point)
                #make sure the direction of the contour doesn't switch - make sure start point is close to prev start point
                if prev_start_point is not None:
                    current_start_point = midline_polyline[0]
                    current_end_point = midline_polyline[-1]
                    if np.linalg.norm(prev_start_point - current_end_point) < np.linalg.norm(prev_start_point - current_start_point):
                        midline_polyline = np.flip(midline_polyline, axis=0)
                prev_start_point = midline_polyline[0]

                #evenly spaced points along the midline
                midline_points = generate_evenly_spaced_points(midline_polyline, midline_segments)
                           
                # Loop through midline points
                for order, point in enumerate(midline_points[1:-1], start=1):
                    x, y = int(round(point[0])), int(round(point[1]))
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                    # Calculate angle formed by current point and neighboring points
                    prev_point = midline_points[order - 1]
                    next_point = midline_points[order + 1]
                    point_angle = three_point_angle(point, next_point, prev_point)

                    # Find orthogonal endpoints
                    start, end = find_orthogonal_endpoints(prev_point, next_point, point, ortho_length)

                    # Draw the orthogonal line segments
                    if start[1] is not None and end[1] is not None:
                        
                        # Draw line on the output image
                        cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (0, 0, 255), 1)

                        # Extract pixel intensity profile along the orthogonal line segment
                        x_values = np.linspace(int(start[0]), int(end[0]), ortho_length)
                        y_values = np.linspace(int(start[1]), int(end[1]), ortho_length)
                        pixel_intensity_values = gray_raw[y_values.astype(int), x_values.astype(int)]

                        # Write to csv
                        for position, intensity in zip(np.linspace(0, 1, len(pixel_intensity_values), endpoint=False), pixel_intensity_values):
                            csv_writer.writerow([frame_number, order, point[0], point[1], point_angle, position, intensity])

            frame_number += 1

        # Display the output frame by frame
        cv2.imshow('output', frame)
        if cv2.waitKey(30) == ord('q'):
            break

        out.write(frame)

    out.release()