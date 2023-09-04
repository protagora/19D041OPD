import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

def save_as_images(data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the DataFrame rows
    for i, row in enumerate(data.itertuples()):
        # Use index name as directory path to store image in
        directoryName = row[0]

        if not os.path.exists(os.path.join(output_folder, directoryName)) or not os.path.isdir(os.path.join(output_folder, directoryName)):
            os.makedirs(os.path.join(output_folder, directoryName))

        image_data = np.array(row[1:]).reshape(int(np.sqrt(len(row[1:]))), -1)

        # Convert the data to an image
        image = Image.fromarray(np.uint8(image_data * 255), 'L')

        # Save the image
        image.save(os.path.join(output_folder, directoryName, f'image_{i}.png'))

# Read the Parquet file
data = pq.read_table('output/recurrence_features.parquet').to_pandas()

# Save rows as images
save_as_images(data, 'recurrence_features_classes')
