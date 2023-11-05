from elpv.utils.elpv_reader import load_dataset

import cv2 as cv
import numpy as np

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


def load_and_preprocess_dataset():
    images, probs, types = load_dataset()
    images = images.astype("float32") / 255

    images = contrast_stretch(images)
    

    images, probs, types = expand_dataset(images, probs, types)

    plt.subplot(2,2,1)
    plt.imshow(cv.cvtColor(images[254*4], cv.COLOR_BGR2RGB))
    plt.subplot(2,2,2)
    plt.imshow(cv.cvtColor(images[132*4+2], cv.COLOR_BGR2RGB))
    plt.subplot(2,2,3)
    plt.imshow(cv.cvtColor(images[2*4], cv.COLOR_BGR2RGB))
    plt.subplot(2,2,4)
    plt.imshow(cv.cvtColor(images[893*4+1], cv.COLOR_BGR2RGB))
    plt.show()

    mono_images = images[types == "mono"]
    mono_probs = probs[types == "mono"]

    poly_images = images[types == "poly"]
    poly_probs = probs[types == "poly"]

    probs_oh = probs
    probs_label_encoder = LabelEncoder()
    probs_oh = probs_label_encoder.fit_transform(probs_oh)
    probs_oh = keras.utils.to_categorical(probs_oh)

    images_3 = np.dstack([images] * 3)
    images_3 = np.reshape(images_3, (-1, 300, 300, 3))

    X_train, X_test, y_train, y_test = train_test_split(images_3, probs_oh, test_size=0.25, random_state=50)

def contrast_stretch(imgs):
    for i in range(0, imgs.shape[0]):
        imgs[i] = calc_contrast_stretch(imgs[i], imgs[i].min(), imgs[i].max())

    return imgs

def calc_contrast_stretch(img, min_pix_val_in, max_pix_val_in):
    ''' 
    takes an img channel, min and max value (of that channel or a reference channel)
    returns a contrast stretched version of the channel
    '''

    min_pix_val_out = 0         #a
    max_pix_val_out = 1         #b
    
    range_multiplier = (max_pix_val_out - min_pix_val_out)/(max_pix_val_in - min_pix_val_in)
    img_out = (img + (-(min_pix_val_in))) * range_multiplier + min_pix_val_out
    img_out = np.clip(img_out, min_pix_val_out, max_pix_val_out)
    #img_out = img_out.astype('float32') 
    return img_out

def expand_dataset(imgs, probs, types):
    orig_size = imgs.shape[0]

    imgs = imgs.repeat(4, axis=0)
    probs  = probs.repeat(4)
    types  = types.repeat(4)

    for i in range(0, orig_size):
        i_tmp = 4*i;
        imgs[i_tmp + 1] = np.flip(imgs[i_tmp], 0)                                                   #vertical flip
        imgs[i_tmp + 2] = np.flip(imgs[i_tmp + 1], 1)                                               #vertical and horizontal flip
        imgs[i_tmp + 3] = np.flip(imgs[i_tmp], 1)                                                   #horizontal flip

    return imgs, probs, types
    



load_and_preprocess_dataset()