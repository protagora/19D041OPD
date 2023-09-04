import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the data
data = pd.read_parquet(os.path.join('output', 'recurrence_features.parquet'))

# Prepare the labels and features
le = LabelEncoder()
labels = le.fit_transform(data.index)
features = data.values

# Reshape the features for CNN
n_features = features.shape[1]
side_length = 225  # Change to your requirement
features = features.reshape((-1, side_length, side_length, 1))

# One-hot encode labels
labels = to_categorical(labels)

input_shape = (side_length, side_length, 1)
num_classes = labels.shape[1]
batch_size = 32  # Adjust based on your requirement
epochs = 10  # Adjust based on your requirement

# Create a stratified k fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# To hold the accuracy for each fold
fold_accuracy = []

# K-Fold Cross Validation
for train_index, val_index in skf.split(features, np.argmax(labels, axis=1)):
    x_train, x_val = features[train_index], features[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.0001),
                metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping])

    # Plot accuracy for both training and validation sets
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Append the accuracy of this fold
    fold_accuracy.append(history.history['val_accuracy'][-1])

print('5-Fold Cross Validation accuracy: ', np.mean(fold_accuracy))
