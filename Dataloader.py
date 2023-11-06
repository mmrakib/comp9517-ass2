from elpv.utils.elpv_reader import load_dataset

import cv2 as cv
import numpy as np
import math

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from pyBench import timefunc

@timefunc
def load_and_preprocess_dataset():
    images, probs, types = load_dataset()
    images = images.astype("float32") / 255

    images = contrast_stretch(images)
    
    images = remove_cell_wires(images)
    images = np.float32(images)

    probs = LabelEncoder().fit_transform(probs)
    probs = keras.utils.to_categorical(probs)
    
    images_3chan = np.zeros([images.shape[0], images.shape[1], images.shape[2], 3])
    for i in range(0,images.shape[0]):
        images_3chan[i] = cv.cvtColor(images[i], cv.COLOR_GRAY2BGR)
    images = images_3chan

    mono_images = images[types == "mono"]
    mono_probs = probs[types == "mono"]

    poly_images = images[types == "poly"]
    poly_probs = probs[types == "poly"]

    labels_mono = np.concatenate((mono_probs, np.full([mono_probs.shape[0],4], "mono")),axis=1)
    labels_poly = np.concatenate((poly_probs, np.full([poly_probs.shape[0],4], "poly")),axis=1)

    train_m_imgs, test_m_imgs, train_m_labs, test_m_labs = \
            train_test_split(mono_images, labels_mono, test_size=0.25, random_state=0, shuffle=False)
    train_p_imgs, test_p_imgs, train_p_labs, test_p_labs = \
            train_test_split(poly_images, labels_poly, test_size=0.25, random_state=0, shuffle=False)
    
    train_imgs = np.concatenate((train_m_imgs, train_p_imgs), axis=0) 
    train_labs = np.concatenate((train_m_labs, train_p_labs), axis=0)
    test_imgs  = np.concatenate((test_m_imgs,  test_p_imgs),  axis=0) 
    test_labs  = np.concatenate((test_m_labs,  test_p_labs),  axis=0)


    train_imgs, train_labs = expand_dataset(train_imgs, train_labs)
    test_imgs, test_labs   = expand_dataset(test_imgs, test_labs)

    rand_seed1 = np.random.randint(1, 2147483647)
    rand_seed2 = np.random.randint(1, 2147483647)
    np.random.seed(rand_seed1)
    np.random.shuffle(train_imgs)
    np.random.seed(rand_seed1)
    np.random.shuffle(train_labs)
    np.random.seed(rand_seed2)
    np.random.shuffle(test_imgs)
    np.random.seed(rand_seed2)
    np.random.shuffle(test_labs)

    train_probs = train_labs[:,0]
    train_types = train_labs[:,1]
    test_probs  = test_labs[:,0]
    test_types  = test_labs[:,1]

    return train_imgs, train_probs, train_types, test_imgs, test_probs, test_types



@timefunc
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

@timefunc
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

@timefunc
def expand_dataset(imgs, labs):                                                                                 #augmentation
    orig_size = imgs.shape[0]

    imgs = imgs.repeat(4, axis=0)
    labs  = labs.repeat(4, axis=0)

    for i in range(0, orig_size):
        i_tmp = 4*i
        imgs[i_tmp + 1] = np.flip(imgs[i_tmp], 0)                                                   #vertical flip
        imgs[i_tmp + 2] = np.flip(imgs[i_tmp + 1], 1)                                               #vertical and horizontal flip
        imgs[i_tmp + 3] = np.flip(imgs[i_tmp], 1)                                                   #horizontal flip

    return imgs, labs
    



load_and_preprocess_dataset()