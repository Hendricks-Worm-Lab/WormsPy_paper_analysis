import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import seaborn as sns

# Set the style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Myriad Pro']
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

# Directory containing CSV files
directory = 'aligned'

# List to store dataframes
dataframes = []

# Read all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        dataframes.append(df)

# Define smoothing window
window_size = 1

# Lists to store processed data
truncated_traces = []
truncated_times = []

# Process each dataframe
for df in dataframes:
    # Find the index closest to aligned_time = 0
    zero_index = (df['aligned_time'] - 0).abs().idxmin()
    
    # Calculate start and end indices for truncation
    start_idx = max(0, zero_index - 100)
    end_idx = min(len(df), zero_index + 51)  # +51 to include the 50th point
    
    # Extract the truncated portion of the dataframe
    df_truncated = df.iloc[start_idx:end_idx]
    
    # Apply smoothing
    smoothed = df_truncated['a'].rolling(window=window_size, min_periods=1).mean()
    truncated_traces.append(smoothed.values)
    truncated_times.append(df_truncated['aligned_time'].values)

# Ensure all traces have the same length by finding the minimum length
min_length = min(len(arr) for arr in truncated_traces)
truncated_traces = [arr[:min_length] for arr in truncated_traces]
truncated_times = [arr[:min_length] for arr in truncated_times]

# Convert to numpy arrays
traces_array = np.array(truncated_traces)
# Use the average of all time arrays to ensure consistency
mean_time = np.mean(truncated_times, axis=0)

# Compute the mean trace and standard error
mean_trace = np.mean(traces_array, axis=0)
std_error = np.std(traces_array, axis=0, ddof=1) / np.sqrt(traces_array.shape[0])

# Create figure with appropriate dimensions for a journal column
fig, ax = plt.subplots(figsize=(5, 4))  # typical column width

# Plot individual traces in background (optional)
# for i in range(traces_array.shape[0]):
#     ax.plot(mean_time, traces_array[i], color='blue', alpha=0.1, linewidth=0.5)

# Plot mean trace and standard error
ax.plot(mean_time, mean_trace, color='#1f77b4', linewidth=1.5, label=f'ASH recordings (n={traces_array.shape[0]})')
ax.fill_between(mean_time, 
                mean_trace - std_error, 
                mean_trace + std_error,
                color='#1f77b4', 
                alpha=0.2)

# Add vertical line at t=0
ax.axvline(x=0, color='#d62728', linestyle='--', linewidth=1, label='Reversal onset')

# Improve axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(width=0.5)

# Add minor ticks
ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.tick_params(which='minor', length=2)
ax.tick_params(which='major', length=4)

# Label axes with units
ax.set_xlabel('Time relative to reversal (s)', fontweight='regular')
ax.set_ylabel('ASH $\Delta$F/F', fontweight='regular')

# Add title and legend with custom styling
ax.set_title('ASH Activity Aligned to Reversal Onset', pad=10)
ax.legend(frameon=False, loc='upper right')

# Adjust y-axis limits with a bit of padding to ensure error bars are visible
y_min, y_max = ax.get_ylim()
y_range = y_max - y_min
ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

# Add horizontal bar to show something like stimulus duration (if applicable)
# ax.plot([start_time, end_time], [y_min - 0.1 * y_range, y_min - 0.1 * y_range], 
#        color='black', linewidth=2, clip_on=False)
# ax.text((start_time + end_time)/2, y_min - 0.15 * y_range, 'Stimulus', 
#        ha='center', va='top')

# Add statistics or annotations if needed
max_point = np.argmax(mean_trace)
ax.annotate(f"Peak: {mean_trace[max_point]:.2f}", 
           xy=(mean_time[max_point], mean_trace[max_point]),
           xytext=(mean_time[max_point]+2, mean_trace[max_point]),
           arrowprops=dict(facecolor='black', width=0.5, headwidth=4, headlength=4))

plt.tight_layout()

# Save as high-resolution vector graphic and PNG
plt.savefig('ASH_reversal_activity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('ASH_reversal_activity.png', dpi=300, bbox_inches='tight')

plt.show()