from tensorflow import keras
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def initialize_model():
    model = keras.models.Sequential()

    #model.add(Hist(1))
    model.add(keras.layers.Dense(4))
    model.add(keras.layers.Softmax())

    return model

def train_model(model, x_train, y_train, optimizer="adam", batch_size = 16, epochs = 100, validation_split = 0.25):
    y_train = LabelEncoder().fit_transform(y_train)
    y_train = keras.utils.to_categorical(y_train)

    print(x_train.shape)
    x_train_h = np.zeros([x_train.shape[0], 256])
    for i in range(0,x_train.shape[0]):
        x_train_h[i] = cv.calcHist(x_train[i],[0],None,[256],[0,1]).reshape([256])

    #cv.normalize(x_train_h, x_train_h, alpha=0, beta=256, norm_type=cv.NORM_MINMAX)

    plot_x = list(range(0,256))
    plt.subplot(2,2,1)
    plt.plot(plot_x,x_train_h[50], 'r')
    plt.subplot(2,2,2)
    plt.plot(plot_x,x_train_h[324], 'r')
    plt.subplot(2,2,3)
    plt.plot(plot_x,x_train_h[3000], 'r')
    plt.subplot(2,2,4)
    plt.plot(plot_x,x_train_h[1469], 'r')
    plt.show()
    

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(x_train_h, y_train, epochs = epochs, validation_split = validation_split, 
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
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

class Hist(keras.layers.Layer):
    def __init__(self, buckets=256):
        super().__init__()
        self.buckets  = buckets

    def call(self, inputs):
        a = tf.summary.histogram(name="dense1", data=inputs, buckets=self.buckets)
        print(a, a.shape, type(a))
        return a
    
import Dataloader

train_imgs, train_probs, _, test_imgs, test_probs, _ = \
        Dataloader.load_and_preprocess_dataset()



model = initialize_model()
history = train_model(model, train_imgs, train_probs, epochs=500, batch_size=500)
plot_loss(history)
plot_accuracy(history)