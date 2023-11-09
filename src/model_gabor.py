from tensorflow import keras
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics         as sk_met

from skimage.filters import gabor_kernel, threshold_triangle as thresh, try_all_threshold

import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2 as cv

from pyBench import timefunc

import math

# Pavement crack detection using the Gabor filter
# https://ieeexplore.ieee.org/document/6728529

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

    for j in range(0, 12):
        img_tmp = np.zeros(x_train[0].shape)
        for i in range(1, 8):
            images = filter_images(x_train[j:j+1], i/12)
            img_tmp += images[0]
        img_tmp = np.clip(img_tmp, 0, 1)
        img_tmp=cv.morphologyEx(img_tmp,cv.MORPH_OPEN,np.ones((4,4)))
        plt.subplot(4,6,2*j+1)
        plt.imshow(x_train[j], cmap='grey')
        plt.subplot(4,6,2*j+2)
        plt.imshow(img_tmp, cmap='grey')
    plt.show()

    
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

@timefunc
def filter_images(images, freq):
    kernels = []
    for theta in range(0, 360, 45): # 0 -> 180 in 22.5 steps *2 to be ints
        if theta > 134 and theta < 226: continue
        kernels.append(gabor_kernel(freq, theta/360 * math.pi))
    images_filt = np.zeros(images.shape)
    for i in range(0,images.shape[0]):
        img_inv = (-1*images[i]) + 1
        for kernel in kernels:
            img_tmp = conv(img_inv, kernel)
            img_tmp = img_tmp > thresh(img_tmp)
            if img_tmp.mean() < 0.6: 
                images_filt[i] += img_tmp
            #else:
                #plt.subplot(1,4,1)
                #plt.imshow(img_inv, cmap='grey')
                #plt.subplot(1,4,2)
                #plt.imshow(images_filt[i], cmap='grey')
                #plt.subplot(1,4,3)
                #plt.imshow(img_tmp, cmap='grey')
                #plt.subplot(1,4,4)
                #plt.imshow(img_tmp > thresh(img_tmp), cmap='grey')
                #plt.show()

            
        images_filt[i] = np.clip(images_filt[i], 0, 1)
        #plt.subplot(1,2,1)
        #plt.imshow(images[i], cmap='grey')
        #plt.subplot(1,2,2)
        #plt.imshow(images_filt[i], cmap='grey')
        
        #plt.show()
    return images_filt

def conv(img, kernel):
    #img = (img - img.mean()) / img.std()
    return ndi.convolve(img, np.real(kernel), mode='wrap')
    #return np.sqrt(ndi.convolve(img, np.real(kernel), mode='wrap')**2 +
    #               ndi.convolve(img, np.imag(kernel), mode='wrap')**2)

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
        Dataloader.load_and_preprocess_dataset(out_types="Mono", simple_probs=False, wire_removal="Crop", augment="None", aug_types=["Bright"], crop_pix=40)


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
plt.show()