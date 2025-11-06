import pandas as pd
import numpy as np
import argparse
import os

# Hard-coded parameters
folder = 'angles'
name = '10_angle'
INPUT_FILE = os.path.join(folder, name + ".csv")
OUTPUT_FILE = os.path.join(folder, name + "_smooth.csv")  # Changed from "_terp.csv" to "_smooth.csv"
WINDOW_SIZE = 5  # Changed from THRESHOLD to WINDOW_SIZE

def smooth_with_moving_average(df, columns=['RIA_angle', 'head_angle'], window_size=5):
    """
    Smooth data in the specified columns using a moving average.
    
    Args:
        df: DataFrame containing the data
        columns: List of column names to smooth
        window_size: Size of the moving average window
        
    Returns:
        DataFrame with smoothed values
    """
    # Create a copy of the dataframe
    result = df.copy()
    
    for column in columns:
        # Apply moving average smoothing
        result[column] = df[column].rolling(window=window_size, center=True).mean()
        
        # Handle edge cases (NaN values at the beginning and end due to window size)
        result[column] = result[column].fillna(method='ffill').fillna(method='bfill')
        
        print(f"Applied moving average with window size {window_size} to '{column}' column")
    
    return result

def main():
    try:
        # Read the input CSV
        print(f"Reading data from {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        
        # Validate that required columns exist
        required_columns = ['frame', 'RIA_angle', 'head_angle']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Input CSV is missing required columns: {missing_columns}")
            return
        
        # Process the data
        print(f"Smoothing data with window size = {WINDOW_SIZE}")
        processed_df = smooth_with_moving_average(df, window_size=WINDOW_SIZE)
        
        # Save the processed data
        processed_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Smoothed data saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()