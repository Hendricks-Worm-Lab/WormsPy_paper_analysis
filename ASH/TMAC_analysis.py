import tmac.models as tm
import tmac.preprocessing as tp
import scipy.io as sio
import pickle
import numpy as np
from pathlib import Path
import pandas as pd

# get the path from user input
folder_path = 'ASH_Feb25'
folder_path = Path(folder_path)

CSV_File = folder_path / 'worm6.csv'
tmac_save_path = folder_path / 'worm6'

# load in csv data
if CSV_File.is_file():
    csv_data = pd.read_csv(CSV_File)
    red = csv_data['RFP_raw'].values
    green = csv_data['GCaMP_raw'].values
    sample_rate = 10

# interpolate to get rid of nans in data
red_interp = tp.interpolate_over_nans(red)[0]
green_interp = tp.interpolate_over_nans(green)[0]

# correct for photobleaching by dividing by an exponential fit to the fluorescence
red_corrected = tp.photobleach_correction(red_interp)
green_corrected = tp.photobleach_correction(green_interp)

# run tmac on the red and green channel to extract the activity
trained_variables = tm.tmac_ac(red_corrected, green_corrected)

nan_loc = np.isnan(red) | np.isnan(green)
a_nan = trained_variables['a'].copy()
a_nan[nan_loc] = np.array('nan')

# add the raw red and green to the output variables
trained_variables['r_raw'] = red
trained_variables['g_raw'] = green
trained_variables['r_corrected'] = red_corrected
trained_variables['g_corrected'] = green_corrected
trained_variables['a_nan'] = a_nan
trained_variables['sample_rate'] = sample_rate
# save to matlab format
sio.savemat(tmac_save_path.with_suffix('.mat'), trained_variables)

# save to pickle format for python
pickle_out = open(tmac_save_path.with_suffix('.pkl'), 'wb')
pickle.dump(trained_variables, pickle_out)
pickle_out.close()