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
    
    remove_cell_wires(images)

    images, probs, types = expand_dataset(images, probs, types)

    a = images[2*4][76].mean()
    b = images[893*4+1][76].mean()
    c = images[2*4][76].std()
    d = images[893*4+1][76].std()

    plt.subplot(2,2,1)
    plt.imshow(cv.cvtColor(images[14*4], cv.COLOR_BGR2RGB))
    plt.subplot(2,2,2)
    plt.imshow(cv.cvtColor(images[22*4], cv.COLOR_BGR2RGB))
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

def contrast_stretch(imgs):                                                                                             #contrast stretching
    for i in range(0, imgs.shape[0]):                              
        imgs[i] = calc_contrast_stretch(imgs[i], imgs[i].min()+0.1, imgs[i].max())

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


def remove_cell_wires(imgs):     
    height        = 300                                                                                       #crop out power wires from cell
    height_mid    = 150
    height_3_wire = 50
    height_2_wire = 75

    for i in range(0, 500):#imgs.shape[0]): 
        score_a,height_a = find_wire(imgs[i], 50)
        score_b,height_b = find_wire(imgs[i], 75)
        score_c,height_c = find_wire(imgs[i], 150)
        score_d,height_d = find_wire(imgs[i], 225)
        score_e,height_e = find_wire(imgs[i], 250)
        if((score_a+score_c+score_e)/3 < (score_b+score_d)/2):
            remove_wire(imgs[0], height_a, 6)
            remove_wire(imgs[0], height_c, 6)
            remove_wire(imgs[0], height_e, 6)
        else:
            remove_wire(imgs[0], height_b, 9)
            remove_wire(imgs[0], height_d, 9)
        
    
    # remove wire locs

def find_wire(img, start_height):
    score = -1
    best_i = 0
    for i in range(start_height-5, start_height+5):
        cur_score = img[i].std() * img[i].mean()
        if(cur_score < score or score < 0):
            score = cur_score
            best_i = i
    return score, best_i

def remove_wire(img, height, width):


def expand_dataset(imgs, probs, types):                                                                                 #augmentation
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