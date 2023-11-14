from tensorflow import keras
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics         as sk_met

from skimage.filters import gabor_kernel, threshold_triangle as thresh_t, threshold_mean as thresh_m, try_all_threshold

import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2 as cv

from pyBench import timefunc

from multiprocessing import Pool
import itertools

import math

# Pavement crack detection using the Gabor filter
# https://ieeexplore.ieee.org/document/6728529

def initialize_model(): 
    #vgg19_base = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(240, 280, 3))
    #vgg19_base.trainable = False
    #model = keras.models.Sequential([
    #vgg19_base,
    #keras.layers.GlobalAveragePooling2D(),
    #keras.layers.Flatten(input_shape=vgg19_base.output_shape[1:]),
    #keras.layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
    #keras.layers.Dense(2048, activation='relu', kernel_initializer='he_normal'),
    #keras.layers.Dense(4, activation='softmax')
    #])

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(240, 280, 3)))#240, 250, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(2, activation = "softmax"))

    return model

def train_model(model, x_train, y_train, optimizer="adam", batch_size = 16, epochs = 100, validation_split = 0.25, workers=18):

    y_train = LabelEncoder().fit_transform(y_train)
    y_train = keras.utils.to_categorical(y_train)

    images_split = []
    block_size = math.ceil(x_train.shape[0] / (3*workers))
    for i in range(0, 3*workers+1):
        if i*block_size >= len(x_train):
            break
        images_split.append(x_train[i*block_size:(i+1)*block_size])
        if(x_train.shape[0] < (i+1)*block_size):
            break
    del x_train, block_size # yes im using that much ram :D

    with Pool(workers) as pool:
        images = np.concatenate( pool.map(filter_images, images_split, chunksize=1), axis=0 )
    del images_split
    images = np.moveaxis(images, 1, -1)
    
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(images, y_train, epochs = epochs, validation_split = validation_split, 
                        batch_size = batch_size)
    
    return history

def predict(model, x_test, y_true, workers=18):
    images_split = []
    block_size = math.ceil(x_test.shape[0] / (3*workers))
    for i in range(0, 3*workers+1):
        images_split.append(x_test[i*block_size:(i+1)*block_size])
        if(x_test.shape[0] < (i+1)*block_size):
            break
    del x_test, block_size # yes im using that much ram :D

    with Pool(workers) as pool:
        images = np.concatenate( pool.map(filter_images, images_split, chunksize=1), axis=0 )
    del images_split
    images = np.moveaxis(images, 1, -1)

    y_prediction = model.predict(images)
    y_prediction = np.argmax(y_prediction, axis=1)

    y_true = LabelEncoder().fit_transform(y_true)
    y_true = keras.utils.to_categorical(y_true)
    y_true = np.argmax(y_true, axis=1)
    
    return y_true, y_prediction

def filter_images(images):
    filters = make_filters([0.13, 0.25, 0.35, 0.45, 0.62, 0.70])
    filters_tri  = [0,1,2,3]
    filters_mean = [2] #4

    images_filt = np.zeros((images.shape[0], 3, images.shape[1], images.shape[2])) #len(filters)+2
    max_val = images.max()
    for i in range(0, images.shape[0]):
        img_inv = (-1*images[i]) + 1
        #img_inv = (img_inv - img_inv.mean()) / img_inv.std()

        #images_filt[i][0] = img_inv

        #k = 1
        k_tri = 2
        for j in range(0, len(filters)):
            filter_tri  = j in filters_tri
            filter_mean = j in filters_mean
            img_tri, img_mean = conv(img_inv, filters[j], filter_tri, filter_mean)
            if filter_tri:
                images_filt[i][k_tri] += img_tri
        #    if filter_mean:
        #        images_filt[i][k] = img_mean
        #        k+=1
        #images_filt[i][k_tri] /= len(filters_tri)
        images_filt[i][k_tri] = (np.clip(images_filt[i][k_tri], 0, 1))
        ##images_filt[i][k_tri] = Dataloader.calc_contrast_stretch(images_filt[i][k_tri], 
        ##                                images_filt[i][k_tri].min(), images_filt[i][k_tri].max())
#
        ##for k in [max_val*.1, max_val*.4, max_val*.65, max_val*.8, max_val*.9]:
        ##    images_filt[i][0] +=  img_inv > k
        ##images_filt[i][0] /= 4
#
        ##images_filt[i][0] = images_filt[i][0] * img_inv
        #images_filt[i][1] = images_filt[i][1] * img_inv
        #images_filt[i][2] = images_filt[i][2] * img_inv
        images_filt[i][0] = cv.Canny((img_inv*255).astype(np.uint8), 20, 50)
        images_filt[i][1] = cv.Canny((img_inv*255).astype(np.uint8), 50, 80)

        #plt.subplot(2,2,1)
        #plt.imshow(img_inv, cmap='grey')
        #plt.subplot(2,2,2)
        #plt.imshow(images_filt[i][0], cmap='grey')
        #plt.subplot(2,2,3)
        #plt.imshow(images_filt[i][1], cmap='grey')
        #plt.subplot(2,2,4)
        #plt.imshow(images_filt[i][2], cmap='grey')
        #plt.show()

        #for j in range(0, 5):
        #    plt.subplot(2,3,j+1)
        #    plt.imshow(images_filt[i][j], cmap='grey')
        #plt.show()
    return images_filt

@timefunc
def make_filters(freqs):
    kernels = []
    i = 0
    for freq in freqs:
        kernels.append([])
        for theta in [0, 45, 70, 135]:#[0, 22.5, 45, 135, 157.5]:
            kernels[i].append(np.real(gabor_kernel(freq, theta/180 * math.pi)))
        i+=1
    return kernels

def conv(img, filter, thresh_tri=True, thresh_mean=False):
    img = (img - img.mean()) / img.std()
    img_tri = np.zeros(img.shape)
    img_mean = np.zeros(img.shape)
    for kernel in filter:
        img_tmp = ndi.convolve(img, kernel, mode='wrap')        

        if thresh_tri:
            img_tri_tmp = img_tmp > thresh_t(img_tmp)
            if img_tri_tmp.mean() < 0.50:   # 100% white kernals are not usefull
                img_tri += img_tri_tmp
        if thresh_mean:
            img_mean_tmp = (img_tmp > thresh_m(img_tmp)).astype(np.float64)
            img_mean_tmp = cv.morphologyEx(img_mean_tmp,cv.MORPH_OPEN,np.ones((3,3)))
            if img_mean_tmp.mean() < 0.95:
                img_mean += img_mean_tmp
    if thresh_tri:
        if img_tri.max() == 0:
            img_tri = img_tri_tmp
        else:
            img_tri /= len(filter)
    if thresh_mean:
        if img_mean.max() == 0:
            img_mean = img_mean_tmp
        else:
            img_mean /= len(filter)
            #images_filt[i][7+j] = np.clip(images_filt[i][7+j], 0, 1)

    return img_tri, img_mean

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 5])
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
if __name__ == '__main__':
    train_imgs, train_probs, train_types, test_imgs, test_probs, test_types = \
            Dataloader.load_and_preprocess_dataset(out_types="Poly", simple_probs=True, wire_removal="Crop", augment="None", aug_types=["Flip", "Bright"], crop_pix=10, shuffle=True, balance_probs=0)


    model = initialize_model()
    history = train_model(model, train_imgs, train_probs, epochs=50, batch_size=800, validation_split=0.10)
    plot_loss(history)
    plot_accuracy(history)


    #Predict
    y_true, y_prediction = predict(model, test_imgs, test_probs)
    print("test |",
        " Accuracy:",   round(sk_met.accuracy_score( y_true, y_prediction),5),
        " Precision:",  round(sk_met.precision_score(y_true, y_prediction, average='macro'),5),
        " Recall:",     round(sk_met.recall_score(   y_true, y_prediction, average='macro'),5),
        " F1:",         round(sk_met.f1_score(       y_true, y_prediction, average='macro'),5))
    sk_met.ConfusionMatrixDisplay(sk_met.confusion_matrix(y_true, y_prediction)).plot()
    plt.show()