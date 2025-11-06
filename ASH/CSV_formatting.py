import pandas as pd

# Read in the CSV files
df1 = pd.read_csv('ASH_Feb25/worm6R.csv') # RFP
df2 = pd.read_csv('ASH_Feb25/worm6G.csv') #GCaMP

# Extract the '25px sum' column from each
rfp_raw = df1['25px_sum']
gcamp_raw = df2['25px_sum']

# Combine the two columns in one dataframe
combined_df = pd.DataFrame({
    'RFP_raw': rfp_raw,
    'GCaMP_raw': gcamp_raw
})
# Replace zero values with NaN
combined_df.replace(0, pd.NA, inplace=True)
# Save the combined dataframe to a new CSV file
combined_df.to_csv('ASH_Feb25/worm6.csv', index=False)