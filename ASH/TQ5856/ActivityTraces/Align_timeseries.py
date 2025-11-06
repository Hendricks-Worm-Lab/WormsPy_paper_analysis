import pandas as pd
import argparse

def align_time_series(df, time_col, align_point):
    """
    Aligns a time series DataFrame by shifting the specified time column so that the
    provided alignment point becomes zero.

    Parameters:
      df (pd.DataFrame): DataFrame containing the time series data.
      time_col (str): Name of the column holding time values.
      align_point (float): The time value that should be set to zero.

    Returns:
      pd.DataFrame: A new DataFrame with an additional column 'aligned_time'.
    """
    df['aligned_time'] = df[time_col] - align_point
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Align multiple time series based on a manually annotated alignment point."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Paths to CSV files containing time series data."
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default="time",
        help="Name of the time column in the CSV files (default: 'time')."
    )
    parser.add_argument(
        "--align_points",
        nargs="+",
        type=float,
        required=True,
        help="Alignment points for each file (the value in the time column to set to zero)."
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_aligned",
        help="Suffix to append to output file names (default: '_aligned')."
    )

    args = parser.parse_args()

    if len(args.files) != len(args.align_points):
        raise ValueError("The number of files must match the number of alignment points provided.")

    for file_path, align_point in zip(args.files, args.align_points):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        if args.time_col not in df.columns:
            print(f"File {file_path} does not contain the column '{args.time_col}'. Skipping.")
            continue

        df_aligned = align_time_series(df, args.time_col, align_point/10)
        output_file = file_path.replace(".csv", f"{args.output_suffix}.csv")
        df_aligned.to_csv(output_file, index=False)
        print(f"Aligned time series saved to {output_file}")

if __name__ == "__main__":
    main()
