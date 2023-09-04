import os
import shutil
import json
import numpy as np
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, TimeDistributed, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

EPOCHS = 100
FOLDS = 10

class TimeDistributedImageDataGenerator:
    def __init__(self, inner_generator):
        self.inner_generator = inner_generator

    def __iter__(self):
        return self

    def __next__(self):
        X, y = next(self.inner_generator)
        X = X[:, None]  # Add the time dimension
        return X, y

    def __len__(self):
        return len(self.inner_generator)


def prepare_data():
    if os.path.exists('recurrence_features_classes_copy'):
        shutil.rmtree('recurrence_features_classes_copy')

    shutil.copytree('recurrence_features_classes', 'recurrence_features_classes_copy')

    # Set the directory where the data is currently located
    data_dir = os.path.join('recurrence_features_classes_copy')

    # Set directories where the train and test data will be stored
    train_dir = os.path.join('output', 'temp_train')
    test_dir = os.path.join('output', 'temp_test')

    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of all classes
    classes = os.listdir(data_dir)
    classes = [item for item in classes if not item.startswith('.')]

    print(f'Classes: {len(classes)}')

    # Loop over each class and split the data
    for class_name in classes:
        # Create train/test directories for the class
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Get list of all images in the class directory
        image_files = os.listdir(os.path.join(data_dir, class_name))

        # Shuffle the list of images
        np.random.shuffle(image_files)

        # Split the list of images into train and test
        split_point = int(len(image_files) * 0.8)  # 80% for training, adjust as needed
        train_files = image_files[:split_point]
        test_files = image_files[split_point:]

        # print(f'Class: {class_name}: {len(train_files)} / {len(test_files)}')

        # Move the train files to the train directory
        for file_name in train_files:
            shutil.move(os.path.join(data_dir, class_name, file_name), os.path.join(train_dir, class_name, file_name))

        # Move the test files to the test directory
        for file_name in test_files:
            shutil.move(os.path.join(data_dir, class_name, file_name), os.path.join(test_dir, class_name, file_name))

def train_model(fold=None):
    # Set directories where the train and test data are stored
    train_dir = os.path.join('output', 'temp_train')
    test_dir = os.path.join('output', 'temp_test')

    # Get list of all classes
    classes = os.listdir(train_dir)  # Classes are folder names in the train directory
    num_classes = len(classes)

    # Set up image generator
    datagen = ImageDataGenerator()  # preprocess_input removed because ConvLSTM2D does not require it

    # Generator for training data
    # train_generator = datagen.flow_from_directory(train_dir, target_size=(260, 260), batch_size=32, class_mode='categorical', color_mode='grayscale')

    # Generator for testing data
    # test_generator = datagen.flow_from_directory(test_dir, target_size=(260, 260), batch_size=32, class_mode='categorical', color_mode='grayscale')

    train_generator = TimeDistributedImageDataGenerator(datagen.flow_from_directory(train_dir, target_size=(225, 225), batch_size=32, class_mode='categorical', color_mode='grayscale'))
    test_generator = TimeDistributedImageDataGenerator(datagen.flow_from_directory(test_dir, target_size=(225, 225), batch_size=32, class_mode='categorical', color_mode='grayscale'))

    # Define the model
    model = Sequential()
    model.add(TimeDistributed(ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same', return_sequences=True), 
                              input_shape=(None, 225, 225, 1)))
    model.add(BatchNormalization())
    
    model.add(TimeDistributed(ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same', return_sequences=True)))
    model.add(BatchNormalization())
    
    model.add(TimeDistributed(ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same', return_sequences=False)))
    model.add(BatchNormalization())

    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)

    # Save history to file
    with open('dijagram_1_data.json', 'w') as handle:
        history_dict = {key: list(values) for key, values in history.history.items()}
        json.dump(history_dict, handle)

    # Save the trained model
    if fold is None:
        fold = 0
    
    model.save(f'trained_model_{fold}.h5')

    # Plot performance
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy for fold ' + str(fold))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Accuracy_plot_fold_' + str(fold) + '.png')

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss for fold ' + str(fold))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Loss_plot_fold_' + str(fold) + '.png')

    # Cleanup and prepare for next fold
    shutil.rmtree(train_dir)
    shutil.rmtree(test_dir)


if "__main__" == __name__:
    for fold in range(1, FOLDS + 1):
        prepare_data()
        train_model(fold)
