from tensorflow import keras

from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def initialize_model():
    model = keras.models.Sequential()

    model.add(Hist(1))
    model.add(keras.layers.Dense(4))
    model.add(keras.layers.Softmax())

    return model

def train_model(model, x_train, y_train, optimizer="adam", batch_size = 16, epochs = 100, validation_split = 0.25):
    y_train = LabelEncoder().fit_transform(y_train)
    y_train = keras.utils.to_categorical(y_train)

    print(x_train.shape)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs = epochs, validation_split = validation_split, 
                        batch_size = batch_size)
    #model.save(path)
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

class Hist(keras.layers.Layer):
    def __init__(self, channels=0, buckets=256):
        super().__init__()
        self.channels = channels
        self.buckets  = buckets

    def call(self, inputs):
        print([inputs])
        a=1
        return cv.calcHist([inputs], channels=self.channels-1, mask=None,
                            histSize=[256], ranges=[0,1])
    
import Dataloader

train_imgs, train_probs, _, test_imgs, test_probs, _ =\
        Dataloader.load_and_preprocess_dataset()



model = initialize_model()
history = train_model(model, train_imgs, train_probs, epochs=1)
plot_loss(history)
plot_accuracy(history)