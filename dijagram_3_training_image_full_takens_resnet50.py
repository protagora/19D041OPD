import os
import shutil
import json
import numpy as np
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt

EPOCHS = 100
FOLDS = 10

def prepare_data():
    if os.path.exists('takens_embedding_features_classes_copy'):
        shutil.rmtree('takens_embedding_features_classes_copy')

    shutil.copytree('takens_embedding_features_classes', 'takens_embedding_features_classes_copy')

    data_dir = os.path.join('takens_embedding_features_classes_copy')

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

        split_point = int(len(image_files) * 0.8)  # 80% for training, adjust as needed
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
    # classes = [item for item in classes if not item.startswith('.')]
    num_classes = len(classes)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

    test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

    base_model = ResNet50(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)

    with open('dijagram_1_data.json', 'w') as handle:
        history_dict = {key: list(values) for key, values in history.history.items()}
        json.dump(history_dict, handle)

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

def clean_training_paths():
    PATH = 'output'
    DIRECTORY_TEST = 'temp_test'
    DIRECTORY_TRAIN = 'temp_train'

    paths = [
        os.path.join(PATH, DIRECTORY_TEST),
        os.path.join(PATH, DIRECTORY_TRAIN),
    ]

    for path in paths:
        shutil.rmtree(path)
        os.makedirs(path)

if "__main__" == __name__:
    for fold in range(1, FOLDS + 1):
        clean_training_paths()
        prepare_data()
        train_model(fold)
