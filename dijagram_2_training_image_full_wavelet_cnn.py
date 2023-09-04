import os
import shutil
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

EPOCHS = 100
FOLDS = 10
TRANSFORM = 'contwavelet'

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def prepare_data():
    if os.path.exists(f'{TRANSFORM}_features_classes_copy'):
        shutil.rmtree(f'{TRANSFORM}_features_classes_copy')

    shutil.copytree(f'{TRANSFORM}_features_classes', f'{TRANSFORM}_features_classes_copy')

    data_dir = os.path.join(f'{TRANSFORM}_features_classes_copy')
    train_dir = os.path.join('output', 'temp_train')
    test_dir = os.path.join('output', 'temp_test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = os.listdir(data_dir)
    classes = [item for item in classes if not item.startswith('.')]
    print(f'Classes: {len(classes)}')

    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        image_files = os.listdir(os.path.join(data_dir, class_name))
        np.random.shuffle(image_files)
        split_point = int(len(image_files) * 0.8)
        train_files = image_files[:split_point]
        test_files = image_files[split_point:]

        for file_name in train_files:
            shutil.move(os.path.join(data_dir, class_name, file_name), os.path.join(train_dir, class_name, file_name))

        for file_name in test_files:
            shutil.move(os.path.join(data_dir, class_name, file_name), os.path.join(test_dir, class_name, file_name))

def train_model(fold=None):
    train_dir = os.path.join('output', 'temp_train')
    test_dir = os.path.join('output', 'temp_test')

    if not (os.path.exists(train_dir) and os.path.isdir(train_dir)):
        os.makedirs(train_dir)
    
    if not (os.path.exists(test_dir) and os.path.isdir(test_dir)):
        os.makedirs(test_dir)

    classes = os.listdir(train_dir)
    num_classes = len(classes)

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(train_dir, target_size=(225, 225), batch_size=32, class_mode='categorical')
    test_generator = datagen.flow_from_directory(test_dir, target_size=(225, 225), batch_size=32, class_mode='categorical')

    model = create_model((225, 225, 3), num_classes)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)

    with open(f'dijagram_{TRANSFORM}_data.json', 'w') as handle:
        json.dump(history.history, handle)

    if fold is None:
        fold = 0

    model.save(f'trained_model_{fold}.h5')

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

    shutil.rmtree(train_dir)
    shutil.rmtree(test_dir)

if "__main__" == __name__:
    for fold in range(1, FOLDS + 1):
        prepare_data()
        train_model(fold)
