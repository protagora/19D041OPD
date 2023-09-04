import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pywt

from library import list_flac_files, load_file

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def generate_cwt(signal, sample_rate):
    scales = np.arange(1, 101)
    wavelet = 'morl' # Morlet wavelet
    cwtmatr, freqs = pywt.cwt(signal, scales, wavelet, 1.0/sample_rate)
    return cwtmatr

def save_as_parquet(flac_files, output_path):
    data_list = [] # This will store each transformed array with a label
    labels_list = [] # This will store the labels

    for index, flac_file in enumerate(flac_files):
        # Get label from the filename (Before K1, K2 or K3)
        label = os.path.basename(flac_file).split('_r')[0]
        print(f'Processing ({index} / {label}): {flac_file}' + ' '.join([' '] * 25), end='\r')

        # Read your CSV data
        data, sampling_frequency = load_file(flac_file)
        data = pd.DataFrame(data)
        data = data.iloc[::1000, :]
        data.columns = ['voltage', 'current']

        # Generate the CWT
        cwtmatr = generate_cwt(data['current'].values, sampling_frequency)

        # Normalize the CWT
        normalized_cwt = normalize_data(cwtmatr)

        # Flatten and store each transformed array into the data_list
        data_list.append(normalized_cwt.flatten())
        labels_list.append(label)

    # Convert list to Pandas DataFrame with labels as index
    data_df = pd.DataFrame(data_list, index=labels_list)

    # Convert DataFrame to Parquet file
    table = pa.Table.from_pandas(data_df)
    pq.write_table(table, os.path.join(output_path, 'contwavelet_features.parquet'))

# Path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = '..'
DATASET_DIRECTORY = 'WHITED'
DATA_DIRECTORY = 'WHITEDv1.1'
GLOB_PATTERN = '*.flac'
OUTPUT_PATH = 'output'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Compute path
path = os.path.join(BASE_PATH, DATASET_DIRECTORY, DATA_DIRECTORY)
flac_files = list_flac_files(path)

# Save as parquet
save_as_parquet(flac_files, OUTPUT_PATH)
