import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def convert_to_polar_binned(csv_path, output_path=None):
    """
    Convert head angle data to polar coordinates and bin into 10-degree sections
    
    Args:
        csv_path: Path to input CSV file with head_angle column
        output_path: Path to save the output CSV file (optional)
        
    Returns:
        DataFrame with the converted data
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Calculate derivative of head_angle (dhead/dt)
    head_angle = df['head_angle'].values
    dhead_dt = np.gradient(head_angle)
    
    # Calculate polar angle from (head_angle, dhead_dt) using atan2
    # atan2(y, x) - head_angle is x, dhead_dt is y
    theta = np.arctan2(dhead_dt, head_angle)
    
    # Ensure angles are in [0, 2*pi]
    theta = (theta + 2*np.pi) % (2*np.pi)
    
    # Convert to degrees for binning
    theta_deg = theta * 180/np.pi
    
    # Create bins from 0 to 360 degrees with 10-degree width
    bin_size = 10
    bin_edges = np.arange(0, 361, bin_size)
    bin_centers = bin_edges[:-1] + bin_size/2
    
    # Find which bin each angle belongs to
    bin_indices = np.searchsorted(bin_edges, theta_deg, side='right') - 1
    
    # Handle edge case where theta_deg is exactly 360 degrees
    bin_indices[bin_indices == len(bin_edges)-1] = 0
    
    # Get the binned angle (center of the bin)
    binned_theta_deg = bin_centers[bin_indices]
    binned_theta_rad = binned_theta_deg * np.pi/180
    
    # Calculate x,y coordinates on the unit circle for the binned angles
    polar_x = np.cos(binned_theta_rad)
    polar_y = np.sin(binned_theta_rad)
    
    # Create a dataframe with the results
    result_df = pd.DataFrame({
        'frame': np.arange(len(head_angle)),
        'head_angle': head_angle,
        'dhead_dt': dhead_dt,
        'theta_rad': theta,
        'theta_deg': theta_deg,
        'bin_index': bin_indices,
        'binned_theta_deg': binned_theta_deg,
        'binned_theta_rad': binned_theta_rad,
        'polar_x': polar_x,
        'polar_y': polar_y
    })
    
    # Plot the data
    plt.figure(figsize=(15, 10))
    
    # Plot the original head angle
    plt.subplot(2, 2, 1)
    plt.plot(result_df['frame'], result_df['head_angle'])
    plt.title('Original Head Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle')
    plt.grid(True)
    
    # Plot phase space (head_angle vs dhead_dt)
    plt.subplot(2, 2, 2)
    plt.scatter(result_df['head_angle'], result_df['dhead_dt'], 
                c=result_df['frame'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Frame')
    plt.title('Phase Space: Head Angle vs. dHead/dt')
    plt.xlabel('Head Angle')
    plt.ylabel('dHead/dt')
    plt.grid(True)
    
    # Plot points on the unit circle
    plt.subplot(2, 2, 3)
    # Draw unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3)
    
    # Plot binned points
    sc = plt.scatter(result_df['polar_x'], result_df['polar_y'], 
                    c=result_df['frame'], cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Frame')
    plt.title('Binned Polar Coordinates on Unit Circle')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    
    # Plot binned angles over time
    plt.subplot(2, 2, 4)
    plt.scatter(result_df['frame'], result_df['binned_theta_deg'], 
                c=result_df['frame'], cmap='viridis', alpha=0.7)
    plt.title('Binned Polar Angles')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.ylim(0, 360)
    
    plt.tight_layout()
    plt.show()
    
    # Save to a new CSV file
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    
    return result_df

if __name__ == "__main__":
    # Replace with actual path to your input CSV file
    input_path = "angles/1_angle_smooth.csv"  # Update this with your file path
    output_path = "angles/1_polar.csv"
    
    result_df = convert_to_polar_binned(input_path, output_path)