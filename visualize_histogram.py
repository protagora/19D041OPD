import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIRECTORY = 'output'

featureFileNames = [
    'contwavelet_features.parquet',
    'melcepstrum_features.parquet',
    'recurrence_features.parquet',
    'spectrogram_features.parquet',
]

for featureFileName in featureFileNames:
    # Load the data
    data = pd.read_parquet(os.path.join(OUTPUT_DIRECTORY, featureFileName))

    # Flatten the feature data
    features_flattened = data.values.flatten()

    # Plot histogram
    plt.hist(features_flattened, bins=50, edgecolor='k')
    plt.title(f'{featureFileName} ({data.shape})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
