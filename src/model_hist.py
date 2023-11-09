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
        Dataloader.load_and_preprocess_dataset(out_types="All", simple_probs=False, wire_removal="Crop", augment="All", aug_types=["Rot", "Bright"])

train_types_n = (train_types == "mono")

model50 = initialize_model()
model75 = initialize_model()
model100 = initialize_model()
model200 = initialize_model()
model300 = initialize_model()
model500 = initialize_model()
model650 = initialize_model()
model800 = initialize_model()
model1000 = initialize_model()
history = train_model(model50, train_imgs, train_probs, epochs=50, batch_size=10000)
history = train_model(model75, train_imgs, train_probs, epochs=75, batch_size=10000)
history = train_model(model100, train_imgs, train_probs, epochs=100, batch_size=10000)
history = train_model(model200, train_imgs, train_probs, epochs=200, batch_size=10000)
history = train_model(model300, train_imgs, train_probs, epochs=300, batch_size=10000)
history = train_model(model500, train_imgs, train_probs, epochs=500, batch_size=10000)
history = train_model(model650, train_imgs, train_probs, epochs=650, batch_size=10000)
history = train_model(model800, train_imgs, train_probs, epochs=800, batch_size=10000)
history = train_model(model1000, train_imgs, train_probs, epochs=1000, batch_size=10000)
#plot_loss(history)
#plot_accuracy(history)


#Predict
y_true, y_prediction = predict(model50, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#Predict
y_true, y_prediction = predict(model75, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#Predict
y_true, y_prediction = predict(model100, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#Predict
y_true, y_prediction = predict(model200, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#Predict
y_true, y_prediction = predict(model300, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#Predict
y_true, y_prediction = predict(model500, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#Predict
y_true, y_prediction = predict(model650, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#Predict
y_true, y_prediction = predict(model800, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#Predict
y_true, y_prediction = predict(model1000, test_imgs, test_probs)
print("train |",
    " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
    " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
    " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
    " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
#sk_met.ConfusionMatrixDisplay(sk_met.confusion_matrix(y_true, y_prediction)).plot()