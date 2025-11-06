import os
import pickle
import pandas as pd

def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        if 'a' in data:
            df = pd.DataFrame(data['a'], columns=['a'])
            df['time'] = df.index / 10
            return df
    return None

def save_to_csv(data_frame, output_file):
    data_frame.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_directory = 'ASH_Feb25'
    output_directory = 'ASH_Feb25/ActivityTraces'
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(input_directory, filename)
            df = read_pickle_file(file_path)
            if df is not None:
                output_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.csv")
                save_to_csv(df, output_file)