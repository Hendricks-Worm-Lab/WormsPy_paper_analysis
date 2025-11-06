import pickle

def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    result = {
        'a': data.get('a'),
        'a_nan': data.get('a_nan'),
        'm': data.get('m'),
        'g_raw': data.get('g_raw'),
        'r_raw': data.get('r_raw'),
        'g_corrected': data.get('g_corrected'),
        'r_corrected': data.get('r_corrected'),
        'length_scale_a': data.get('length_scale_a'),
        'length_scale_m': data.get('length_scale_m'),
        'variance_a': data.get('variance_a'),
        'variance_m': data.get('variance_m'),
        'variance_g_noise': data.get('variance_g_noise'),
        'variance_r_noise': data.get('variance_r_noise')
    }
    
    return result

data = load_data('ASH_Feb25/worm3.pkl')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data['a'], label='Activity', alpha=0.7)
plt.plot(data['m'], label='Motion Artefacts', alpha=0.7)

# plt.plot(data['g_raw'], label='g_raw', alpha=0.7)
# plt.plot(data['g_corrected'], label='g_corrected', alpha=0.7)
# plt.plot(data['r_raw'], label='r_raw', alpha=0.7)
# plt.plot(data['r_corrected'], label='r_corrected', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Calcium Trace corrected for photobleaching and motion artefacts')
plt.legend()
plt.show()