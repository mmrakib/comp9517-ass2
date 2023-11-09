from tensorflow import keras
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics         as sk_met

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def initialize_model():
    model = keras.models.Sequential()

    #model.add(Hist(1))
    model.add(tf.keras.Input(shape=(256,)))
    model.add(keras.layers.Dense(40, activation='relu'))
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

    cv.normalize(x_train_h, x_train_h, alpha=0, beta=256, norm_type=cv.NORM_MINMAX)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(x_train_h, y_train, epochs = epochs, validation_split = validation_split, 
                        batch_size = batch_size)
    #model.save(path)
    return history

def predict(model, x_test, y_true):
    x_test_h = np.zeros([x_test.shape[0], 256])
    for i in range(0,x_test.shape[0]):
        x_test_h[i] = cv.calcHist(x_test[i],[0],None,[256],[0,1]).reshape([256])

    y_prediction = model.predict(x_test_h)
    y_prediction = np.argmax(y_prediction, axis=1)

    y_true = LabelEncoder().fit_transform(y_true)
    y_true = keras.utils.to_categorical(y_true)
    y_true = np.argmax(y_true, axis=1)
    
    return y_true, y_prediction

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
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


import Dataloader
train_imgs, train_probs, train_types, test_imgs, test_probs, test_types = \
        Dataloader.load_and_preprocess_dataset(out_types="All", simple_probs=False, wire_removal="Crop", augment="All", channels=3, aug_types=["Flip"])


model = initialize_model()
history = train_model(model, train_imgs, train_probs, epochs=300, batch_size=10000)
#plot_loss(history)
#plot_accuracy(history)


#Predict
y_true, y_prediction = predict(model, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
sk_met.ConfusionMatrixDisplay(sk_met.confusion_matrix(y_true, y_prediction)).plot()