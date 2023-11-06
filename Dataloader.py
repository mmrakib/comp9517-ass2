from elpv.utils.elpv_reader import load_dataset

import cv2 as cv
import numpy as np
import math

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


def load_and_preprocess_dataset():
    images, probs, types = load_dataset()
    images = images.astype("float32") / 255

    images = contrast_stretch(images)
    
    images = remove_cell_wires(images)

    images, probs, types = expand_dataset(images, probs, types)

    images = np.float32(images)
    plt.subplot(2,2,1)
    plt.imshow(cv.cvtColor(images[3*4], cv.COLOR_BGR2RGB))
    plt.subplot(2,2,2)
    plt.imshow(cv.cvtColor(images[70*4], cv.COLOR_BGR2RGB))
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
    return img_out


def remove_cell_wires(imgs):     
    imgs_tmp = np.zeros([imgs.shape[0], 240, 280])
    crop = [10,250, 10,290]

    for i in range(0, imgs.shape[0]): 
        score_a,height_a,width_a = find_wire(imgs[i], 50)
        score_b,height_b,width_b = find_wire(imgs[i], 75)
        score_c,height_c,width_c = find_wire(imgs[i], 150)
        score_d,height_d,width_d = find_wire(imgs[i], 225)
        score_e,height_e,width_e = find_wire(imgs[i], 250)
        if((score_a+score_c+score_e)/3 < (score_b+score_d)/2):
            img_tmp = remove_wire(imgs[i], height_a, width_a)
            img_tmp = remove_wire(img_tmp, height_c-width_a, width_c)
            img_tmp = remove_wire(img_tmp, height_e-(width_a+width_c),width_e)
            imgs_tmp[i] = cv.resize(img_tmp, (300,260))[crop[0]:crop[1], crop[2]:crop[3]]
        else:
            img_tmp = remove_wire(imgs[i], height_b, width_b)
            img_tmp = remove_wire(img_tmp, height_d-width_b, width_d)
            imgs_tmp[i] = cv.resize(img_tmp, (300,270))[crop[0]:crop[1], crop[2]:crop[3]]
    return imgs_tmp

def find_wire(img, start_height):
    test_score = img[start_height].std() * img[start_height].mean()
    if(test_score > 0.1):
        return test_score, start_height, 16

    for i in range(start_height, start_height-10, -1):
        top_point = i
        top_point_score = img[top_point].std() * img[top_point].mean()
        if(top_point_score > 0.12):
            break
    for i in range(start_height, start_height+15):
        bottom_point = i
        bottom_point_score = img[bottom_point].std() * img[bottom_point].mean()
        if (bottom_point_score > top_point_score):
            break
    mid_point = int(round((top_point+bottom_point)/2, 0))
    score = -1
    for i in range(mid_point-3, mid_point+3):
        cur_score = img[i].std() * img[i].mean()
        if(cur_score < score or score < 0):
            score = cur_score
    return score, mid_point, bottom_point - top_point + 1

def remove_wire(img, height, width):
    return np.delete(img, np.s_[height - math.floor(width/2):math.ceil(height + width/2)], 0)
#.7

def expand_dataset(imgs, probs, types):                                                                                 #augmentation
    orig_size = imgs.shape[0]

    imgs = imgs.repeat(4, axis=0)
    probs  = probs.repeat(4)
    types  = types.repeat(4)

    for i in range(0, orig_size):
        i_tmp = 4*i
        imgs[i_tmp + 1] = np.flip(imgs[i_tmp], 0)                                                   #vertical flip
        imgs[i_tmp + 2] = np.flip(imgs[i_tmp + 1], 1)                                               #vertical and horizontal flip
        imgs[i_tmp + 3] = np.flip(imgs[i_tmp], 1)                                                   #horizontal flip

    return imgs, probs, types
    



load_and_preprocess_dataset()