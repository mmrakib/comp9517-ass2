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
    model.add(keras.layers.Dense(100, activation='relu'))
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

def predict(model, x_test):
    x_test_h = np.zeros([x_test.shape[0], 256])
    for i in range(0,x_test.shape[0]):
        x_test_h[i] = cv.calcHist(x_test[i],[0],None,[256],[0,1]).reshape([256])
    return model.predict(x_test_h)

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

class Hist(keras.layers.Layer):
    def __init__(self, buckets=256):
        super().__init__()
        self.buckets  = buckets

    def call(self, inputs):
        a = tf.summary.histogram(name="dense1", data=inputs, buckets=self.buckets)
        print(a, a.shape, type(a))
        return a
    
import Dataloader

train_imgs, train_probs, train_types, test_imgs, test_probs, test_types = \
        Dataloader.load_and_preprocess_dataset(out_types="All", simple_probs=False, wire_removal="Crop", augment="All")

train_types_n = (train_types == "mono")

model = initialize_model()
history = train_model(model, train_imgs, train_probs, epochs=1000, batch_size=10000)
plot_loss(history)
plot_accuracy(history)



from sklearn.metrics import confusion_matrix

#Predict
y_prediction = predict(model, train_imgs)


train_probs = LabelEncoder().fit_transform(train_probs)
train_probs = keras.utils.to_categorical(train_probs)
train_probs = np.argmax(train_probs, axis=1)

y_prediction = np.argmax(y_prediction, axis=1)

print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( train_probs, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(train_probs, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   train_probs, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       train_probs, y_prediction, average='macro'),5))
sk_met.ConfusionMatrixDisplay(sk_met.confusion_matrix(train_probs, y_prediction)).plot()
#Predict
y_prediction = predict(model, test_imgs)


test_probs = LabelEncoder().fit_transform(test_probs)
test_probs = keras.utils.to_categorical(test_probs)
test_probs = np.argmax(test_probs, axis=1)

y_prediction = np.argmax(y_prediction, axis=1)

print("test |",
    " Accuracy:",   round(sk_met.accuracy_score( test_probs, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(test_probs, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   test_probs, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       test_probs, y_prediction, average='macro'),5))
sk_met.ConfusionMatrixDisplay(sk_met.confusion_matrix(test_probs, y_prediction)).plot()
plt.show()



a=1