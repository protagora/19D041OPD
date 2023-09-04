import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy import interpolate

def save_as_images(data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the DataFrame rows
    for i, row in enumerate(data.itertuples()):
        # Use index name as directory path to store image in
        directoryName = row[0]

        if not os.path.exists(os.path.join(output_folder, directoryName)) or not os.path.isdir(os.path.join(output_folder, directoryName)):
            os.makedirs(os.path.join(output_folder, directoryName))

        image_data = np.array(row[1:]).reshape(224, 224)

        # Normalize the image data to 0-255 range
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

        # Convert the data to an image
        image = Image.fromarray(np.uint8(image_data * 255), 'L')

        # Save the image
        image.save(os.path.join(output_folder, directoryName, f'image_{i}.png'))

# Read the Parquet file
data = pd.read_parquet('output/takens_embedding_features.parquet')

# Interpolate the data to the desired size (50176)
new_data = []
labels = []
for row in data.itertuples():
    labels.append(row[0])
    old_row = np.array(row[1:])
    x = np.linspace(0, 1, len(old_row))
    f = interpolate.interp1d(x, old_row)
    xnew = np.linspace(0, 1, 50176)  # size for 224x224 image
    new_row = f(xnew)
    new_data.append(new_row)

new_data = pd.DataFrame(new_data, index=labels)

# Save rows as images
save_as_images(new_data, 'takens_embedding_features_classes')
