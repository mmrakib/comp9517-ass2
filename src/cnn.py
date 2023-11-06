from elpv.utils.elpv_reader import load_dataset
import numpy as np

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import cv2 as cv
import matplotlib.pyplot as plt

def initialize_model(version = "vgg19"):
    if version == "type":
        type_model = keras.models.Sequential()
        type_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))
        type_model.add(keras.layers.MaxPooling2D((2, 2)))
        type_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        type_model.add(keras.layers.MaxPooling2D((2, 2)))
        type_model.add(keras.layers.Flatten())
        type_model.add(keras.layers.Dense(64, activation='relu'))
        type_model.add(keras.layers.Dense(2, activation = "softmax"))

        return type_model
 
    elif version == "vgg19":
        vgg19_base = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(300,300,3))
        vgg19_base.trainable = False

        vgg19_model = keras.models.Sequential([
        vgg19_base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Flatten(input_shape=vgg19_base.output_shape[1:]),
        keras.layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(2048, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(4, activation='softmax')
        ])

        return vgg19_model


def train_model(model, X_train, y_train, path, optimizer = "adam", batch_size = 16, epochs = 100, validation_split = 0.2):
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs = epochs, validation_split = validation_split, batch_size = batch_size)
    model.save(path)
    return history

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0.5, 5])
    plt.legend(loc='lower right')

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='loss')
    plt.plot(history.history['val_accuracy'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 5])
    plt.legend(loc='lower right')