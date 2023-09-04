import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

# Read the parquet file
base_dir = 'output'
feature_file = 'recurrence_features.parquet'
data = pd.read_parquet(os.path.join(base_dir, feature_file))

# Extract labels
labels = data.index

# Identify unique labels and number of features
unique_labels = labels.unique()
n_features = data.shape[1]

# Select a subset of features for visualization. 
# Note: visualizing all features may be impractical if the number of features is large
selected_features = data.columns[0:10]

# Loop over each unique label
for label in unique_labels:
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Feature Distributions for {label}', fontsize=16)

    # Loop over each selected feature
    for i, feature in enumerate(selected_features):
        plt.subplot(2, 5, i+1)
        
        # Select rows where the label is 'label' and column 'feature'
        data_label = data.loc[data.index == label, feature]

        # Plot distribution of this feature for this label
        sns.histplot(data=data_label, kde=False, bins=30)
        plt.title(feature)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
