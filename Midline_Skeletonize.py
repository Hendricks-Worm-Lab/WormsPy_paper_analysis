import cv2
import numpy as np
from skimage.morphology import skeletonize
import os
import csv

#this ensures that the points along the midline are ordered and returns them as connected segments
def order_points_by_nearest_neighbor(points, start_point=None):
    # Convert points to tuples to avoid unhashable type error
    points_as_tuples = [tuple(map(int, p)) for p in points]
    
    if start_point is None:
        start_point = points_as_tuples[0]
    else:
        start_point = tuple(map(int, start_point))

    polyline = [start_point]
    remaining_points = set(points_as_tuples)
    remaining_points.remove(start_point)
    while remaining_points:
        last_point = polyline[-1]
        next_point = min(remaining_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(last_point)))
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

# Calculates angle between vector and horizontal direction using atan2
def calculate_segment_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]  # Note: y increases downward in image coordinates
    
    # Calculate angle in radians with atan2
    angle_rad = np.arctan2(-dy, dx)  # Negative dy because y increases downward
    
    # Convert to degrees (0° = right, counterclockwise positive)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# Find the appropriate head position based on user preference
def find_worm_head(midline_polyline, head_position):
    """
    Find the head position based on user preference.
    head_position: 
        "LEFT" - leftmost point (min x) is the head
        "RIGHT" - rightmost point (max x) is the head
        "TOP" - topmost point (min y) is the head
        "BOTTOM" - bottommost point (max y) is the head
        "AUTO" - use traditional method (farthest from centroid)
        "AUTO_ALT" - use second farthest point from centroid
    """
    if head_position == "LEFT":
        # Find the leftmost point (minimum x-coordinate)
        head_idx = np.argmin(midline_polyline[:, 0])
        return midline_polyline[head_idx]
    elif head_position == "RIGHT":
        # Find the rightmost point (maximum x-coordinate)
        head_idx = np.argmax(midline_polyline[:, 0])
        return midline_polyline[head_idx]
    elif head_position == "TOP":
        # Find the topmost point (minimum y-coordinate) 
        # Note: y increases downward in image coordinates
        head_idx = np.argmin(midline_polyline[:, 1])
        return midline_polyline[head_idx]
    elif head_position == "BOTTOM":
        # Find the bottommost point (maximum y-coordinate)
        head_idx = np.argmax(midline_polyline[:, 1])
        return midline_polyline[head_idx]
    elif head_position == "AUTO_ALT":
        # Calculate centroid
        centroid = np.mean(midline_polyline, axis=0)
        
        # Find endpoints (assuming midline has been ordered already)
        end1 = midline_polyline[0]
        end2 = midline_polyline[-1]
        
        # Calculate distances from centroid
        dist1 = np.sum((end1 - centroid)**2)
        dist2 = np.sum((end2 - centroid)**2)
        
        # Return the endpoint CLOSER to the centroid (opposite of AUTO)
        return end2 if dist1 > dist2 else end1
    else:  # AUTO mode - use centroid method
        # Calculate centroid
        centroid = np.mean(midline_polyline, axis=0)
        
        # Find endpoints (assuming midline has been ordered already)
        end1 = midline_polyline[0]
        end2 = midline_polyline[-1]
        
        # Calculate distances from centroid
        dist1 = np.sum((end1 - centroid)**2)
        dist2 = np.sum((end2 - centroid)**2)
        
        # Return the endpoint farthest from centroid
        return end1 if dist1 > dist2 else end2

# Open the video file
name = 10

# User parameter for head position: "LEFT", "RIGHT", "TOP", "BOTTOM", "AUTO", or "AUTO_ALT"
HEAD_POSITION = "BOTTOM"

in_folder = 'RIA_sinusoidal'
input_filename = os.path.join(in_folder, str(name) + ".avi")
cap = cv2.VideoCapture(input_filename)  # Add your video path here

# Number of points to sample along the midline (user parameter)
NUM_POINTS = 12

# Read the first frame
ret, frame = cap.read()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_folder = 'angles'
output_filename = os.path.join(out_folder, f"{name}_skeleton.avi")

# Check if the output file already exists and delete it
if os.path.exists(output_filename):
    os.remove(output_filename)

out = cv2.VideoWriter(output_filename, fourcc, 10.0, (frame.shape[1], frame.shape[0]))
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

# Create CSV header for both vectors (0-1 and 0-2)
csv_header = ["frame", "RIA_angle", "head_angle"]

csv_filename = os.path.join(out_folder, f"{name}_angle.csv")
with open(csv_filename, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

    frame_number = 0
    prev_head_point = None  # To track head point across frames

    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to gray, median filter, threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Stronger blur for initial smoothing
        blurred_frame = cv2.GaussianBlur(gray, (51, 51), 0)  # Increased kernel size

        # Use binary thresholding instead of adaptive for smoother edges
        _, thresh = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # More aggressive morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Larger kernel
        eroded_image = cv2.erode(thresh, kernel, iterations=2)  # Fewer iterations
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=3)  # More dilations
        processed_frame = cv2.medianBlur(dilated_image, 5)  # Additional smoothing

        # Find largest contour
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find and smooth the worm contour
            worm_contour = max(contours, key=cv2.contourArea)
            
            # Draw worm outline
            # cv2.drawContours(frame, [worm_contour], -1, (0, 255, 0), 1)
            
            # Skeletonize the worm contour to find midline
            worm_binary = np.zeros_like(gray)
            cv2.drawContours(worm_binary, [worm_contour], -1, 255, -1)
            skeleton = skeletonize(worm_binary).astype(np.uint8)
            skeleton_binary = np.zeros_like(skeleton)
            skeleton_binary[skeleton == 1] = 255
            midline_contours, _ = cv2.findContours(skeleton_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(midline_contours) > 0:
                # Find longest midline contour
                midline = max(midline_contours, key=lambda c: cv2.arcLength(c, False))
                
                # Convert the midline contour to a polyline
                midline_polyline = contour_to_polyline(midline)
                
                # Find head position based on user preference
                head_point = find_worm_head(midline_polyline, HEAD_POSITION)
                
                # Find endpoints of the midline (assuming it's a simple curve)
                # These should be the farthest points from each other
                pairwise_distances = np.sum((midline_polyline[:, np.newaxis] - midline_polyline[np.newaxis, :]) ** 2, axis=2)
                np.fill_diagonal(pairwise_distances, -1)
                end1_idx, end2_idx = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
                
                # Calculate which endpoint is closer to our target head_point
                dist_to_end1 = np.sum((midline_polyline[end1_idx] - head_point)**2)
                dist_to_end2 = np.sum((midline_polyline[end2_idx] - head_point)**2)
                
                # Choose the endpoint closest to our desired head position
                head_idx = end1_idx if dist_to_end1 < dist_to_end2 else end2_idx
                
                # Create a new properly ordered midline from the head to the tail
                ordered_midline = []
                remaining_points = list(map(tuple, midline_polyline))
                start_point = tuple(midline_polyline[head_idx])
                
                # Build the midline starting from the head
                ordered_midline.append(start_point)
                remaining_points.remove(start_point)
                
                # Add points one at a time, closest first
                while remaining_points:
                    last_point = ordered_midline[-1]
                    next_point = min(remaining_points, 
                                     key=lambda p: np.sum((np.array(p) - np.array(last_point))**2))
                    ordered_midline.append(next_point)
                    remaining_points.remove(next_point)
                
                # Convert back to numpy array
                midline_polyline = np.array(ordered_midline)
                
                # Draw the head point
                cv2.circle(frame, tuple(midline_polyline[0]), 5, (0, 0, 255), -1)
                cv2.putText(frame, "HEAD", (midline_polyline[0][0]+5, midline_polyline[0][1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Mark tail point
                tail_point = midline_polyline[-1]
                cv2.circle(frame, tuple(tail_point), 4, (255, 0, 0), -1)
                cv2.putText(frame, "TAIL", (tail_point[0]+5, tail_point[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw the midline
                # for i in range(len(midline_polyline) - 1):
                #     cv2.line(frame, tuple(midline_polyline[i]), tuple(midline_polyline[i + 1]), 
                #             (255, 0, 0), 1)
                
                # Sample evenly spaced points along the midline
                sampled_points = generate_evenly_spaced_points(midline_polyline, NUM_POINTS)
                
                # Calculate angles between consecutive points
                angles = []
                for i in range(len(sampled_points) - 1):
                    angle = calculate_segment_angle(sampled_points[i], sampled_points[i + 1])
                    angles.append(angle)
                    
                    # Draw a line between consecutive points
                    cv2.line(frame, tuple(sampled_points[i]), tuple(sampled_points[i + 1]), 
                            (0, 255, 255), 1)
                
                # Mark each sampled point
                for i, point in enumerate(sampled_points):
                    cv2.circle(frame, tuple(point), 3, (0, 165, 255), -1)
                    cv2.putText(frame, str(i), (point[0]+5, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.4, (255, 255, 255), 1)
                
                # Prepare data for CSV - vectors 0-1 and 3-2-0 angle
                if len(sampled_points) >= 4:  # Need at least points 0, 2, and 3
                    # Calculate angle directly between points 0 and 1 (relative to horizontal)
                    angle_0_1 = calculate_segment_angle(sampled_points[0], sampled_points[1])
                    
                    # Calculate angle between vectors 3->2 and 2->0
                    # Vector 3->2
                    vec1 = np.array([sampled_points[2][0] - sampled_points[3][0], 
                                    sampled_points[2][1] - sampled_points[3][1]])
                    # Vector 2->0  
                    vec2 = np.array([sampled_points[0][0] - sampled_points[2][0], 
                                    sampled_points[0][1] - sampled_points[2][1]])
                    
                    # Calculate the angle between the two vectors
                    dot_product = np.dot(vec1, vec2)
                    magnitude1 = np.linalg.norm(vec1)
                    magnitude2 = np.linalg.norm(vec2)
                    
                    # Avoid division by zero
                    head_angle = 0
                    if magnitude1 * magnitude2 > 0:
                        cos_angle = dot_product / (magnitude1 * magnitude2)
                        # Clip to handle floating point errors that might put cos_angle outside [-1, 1]
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        head_angle = np.degrees(np.arccos(cos_angle))
                        
                        # Determine the sign of the angle (positive = counterclockwise)
                        cross_product = np.cross(vec1, vec2)
                        if cross_product < 0:
                            head_angle = -head_angle
                        
                        # Display only the head angle (3->2->0)
                        # Calculate midpoint for text placement
                        midpoint = (sampled_points[2][0], sampled_points[2][1])
                        cv2.putText(frame, f"{int(head_angle)}", 
                                  (midpoint[0]+10, midpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (0, 255, 0), 2)
                        
                        # Draw the vectors used for angle calculation
                        cv2.line(frame, tuple(sampled_points[3]), tuple(sampled_points[2]), 
                                (255, 0, 255), 2)  # Vector 3->2
                        cv2.line(frame, tuple(sampled_points[2]), tuple(sampled_points[0]), 
                                (255, 0, 255), 2)  # Vector 2->0
                    
                    # Create CSV row with frame number and angles
                    csv_row = [frame_number, angle_0_1, head_angle]
                    
                    # Write to CSV
                    csv_writer.writerow(csv_row)

        frame_number += 1

        # Display the output frame
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Worm Midline', frame)
        if cv2.waitKey(30) == ord('q'):
            break

        out.write(frame)

    # Release resources
    out.release()
    cap.release()
    cv2.destroyAllWindows()