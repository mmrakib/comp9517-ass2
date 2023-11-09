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
def load_and_preprocess_dataset(out_probs=[0,1,2,3], simple_probs=False, out_types="All", 
                                wire_removal="Crop", channels=1, augment="All", aug_types=["Flip", "Rot", "Bright"]):
    """
    :param out_probs: [0 = undamaged, 1 = mild, 2 = major, 3 = destroyed]
    :param simple_probs: true = set all probs > 0 to 1 (any damage = 1)
    :param out_types: All, Mono, Poly (only return matching)
    :param wire_removal: Crop, Gray 
    :param channels: 1 keep as 1 channel, 3 convert images to 3 channel
    :param augment: All, Train, Test (which data to augment)
    """
    images, probs, types = load_dataset()
    images = images.astype("float32") / 255

    print(len(probs[probs == 0]), \
            len(probs[probs == 0.3333333333333333]), \
            len(probs[probs == 0.6666666666666666]), \
            len(probs[probs == 1]))

    images, probs, types = reduce_dataset(out_probs, simple_probs, out_types, images, probs, types)

    print(len(probs[probs == 0]), \
            len(probs[probs == 0.3333333333333333]), \
            len(probs[probs == 0.6666666666666666]), \
            len(probs[probs == 1]))

    images = contrast_stretch(images)
    
    
    images = remove_cell_wires(images, wire_removal)
    images = np.float32(images)

    train_imgs, train_probs, train_types, test_imgs, test_probs, test_types =\
            split_t_t_data(images, probs, types, out_types)

    print(len(train_probs[train_probs == 0]), \
            len(train_probs[train_probs == 0.3333333333333333]), \
            len(train_probs[train_probs == 0.6666666666666666]), \
            len(train_probs[train_probs == 1]))
    print(len(test_probs[test_probs == 0]), \
            len(test_probs[test_probs == 0.3333333333333333]), \
            len(test_probs[test_probs == 0.6666666666666666]), \
            len(test_probs[test_probs == 1]))

    if(augment == "All" or augment == "Train"):
        train_imgs, train_probs, train_types = expand_dataset(train_imgs, train_probs, train_types, aug_types)
    if(augment == "All" or augment == "Test"):
        test_imgs, test_probs, test_types    = expand_dataset(test_imgs, test_probs, test_types, aug_types)

    train_imgs, train_probs, train_types, test_imgs, test_probs, test_types =\
            shuffle_set(train_imgs, train_probs, train_types, test_imgs, test_probs, test_types)

    print(len(train_probs[train_probs == 0]), \
            len(train_probs[train_probs == 0.3333333333333333]), \
            len(train_probs[train_probs == 0.6666666666666666]), \
            len(train_probs[train_probs == 1]))
    print(len(test_probs[test_probs == 0]), \
            len(test_probs[test_probs == 0.3333333333333333]), \
            len(test_probs[test_probs == 0.6666666666666666]), \
            len(test_probs[test_probs == 1]))

    if(channels == 3):
        train_imgs = make_3_channel(train_imgs)
        test_imgs = make_3_channel(test_imgs)
        
    return train_imgs, train_probs, train_types, test_imgs, test_probs, test_types

@timefunc
def reduce_dataset(out_probs, simple_probs, out_types, images, probs, types):
    if (0 not in out_probs):
        images = images[probs != 0]
        types = types[probs != 0]
        probs = probs[probs != 0]
    if (1 not in out_probs):
        images = images[probs != 0.3333333333333333]
        types = types[probs != 0.3333333333333333]
        probs = probs[probs != 0.3333333333333333]
    if (2 not in out_probs):
        images = images[probs != 0.6666666666666666]
        types = types[probs != 0.6666666666666666]
        probs = probs[probs != 0.6666666666666666]
    if (3 not in out_probs):
        images = images[probs != 1.0]
        types = types[probs != 1.0]
        probs = probs[probs != 1.0]
    if(simple_probs):
        for i in range(0,probs.shape[0]):
            probs[i] = probs[i] > 0

    if (out_types == "Mono"):
        images = images[types == "mono"]
        probs = probs[types == "mono"]
        types = types[types == "mono"]
    elif (out_types == "Poly"):
        images = images[types == "poly"]
        probs = probs[types == "poly"]
        types = types[types == "poly"]

    return images, probs, types


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
def remove_cell_wires(imgs, wire_removal):     
    crop = 10
    if (wire_removal == "Crop"):
        end_size = [240,250]
    else:
        end_size = [280,280]
    imgs_tmp = np.zeros([imgs.shape[0], end_size[0], end_size[1]])

    for i in range(0, imgs.shape[0]): 
        score_a,height_a,width_a = find_wire(imgs[i], 50)
        score_b,height_b,width_b = find_wire(imgs[i], 75)
        score_c,height_c,width_c = find_wire(imgs[i], 150)
        score_d,height_d,width_d = find_wire(imgs[i], 225)
        score_e,height_e,width_e = find_wire(imgs[i], 250)
        if((score_a+score_c+score_e)/3 < (score_b+score_d)/2):
            img_tmp = remove_wire(imgs[i], height_a, width_a, wire_removal, 0)
            img_tmp = remove_wire(img_tmp, height_c, width_c, wire_removal, width_a)
            img_tmp = remove_wire(img_tmp, height_e, width_e, wire_removal, width_a + width_c)
            imgs_tmp[i] = cv.resize(img_tmp, (end_size[0]+2*crop,end_size[1]+2*crop)\
                                    )[crop:end_size[0]+crop, crop:end_size[1]+crop]
        else:
            img_tmp = remove_wire(imgs[i], height_b, width_b, wire_removal, 0)
            img_tmp = remove_wire(img_tmp, height_d, width_d, wire_removal, width_b)
            imgs_tmp[i] = cv.resize(img_tmp, (end_size[0]+2*crop,end_size[1]+2*crop)\
                                    )[crop:end_size[0]+crop, crop:end_size[1]+crop]
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

def remove_wire(img, height, width, wire_removal, offset):
    if(wire_removal == "Crop"):
        height -= offset
        return np.delete(img, np.s_[height - math.floor(width/2):math.ceil(height + width/2)], 0)
    img[height - math.floor(width/2):math.ceil(height + width/2)] = \
        np.full([width, img.shape[1]], 0.78)
    return img
#.7

@timefunc
def expand_dataset(imgs, probs, types, aug_types):                                                                      #augmentation    
    orig_size = imgs.shape[0]

    end_size = 1
    end_size *= 4 if ("Flip" in aug_types) else 1
    end_size *= 7 if ("Rot"  in aug_types) else 1
    end_size *= 3 if ("Bright"  in aug_types) else 1

    imgs = imgs.repeat(end_size, axis=0)
    probs  = probs.repeat(end_size, axis=0)
    types  = types.repeat(end_size, axis=0)

    if("Flip" in aug_types):
        step = int(end_size/4)
        for i in range(0, orig_size):
            i_tmp = end_size*i
            imgs[i_tmp +   step] = np.flip(imgs[i_tmp], 0)                                               #vertical flip
            imgs[i_tmp + 2*step] = np.flip(imgs[i_tmp + step], 1)                                           #vertical and horizontal flip
            imgs[i_tmp + 3*step] = np.flip(imgs[i_tmp], 1)                                               #horizontal flip
        end_size = int(end_size/4)
    
    if("Rot"  in aug_types):
        step = int(end_size/7)
        image_center = tuple(np.array(imgs[0].shape[1::-1]) / 2)
        for i in range(0, orig_size):
            i_tmp = end_size*i
            rot_mat = cv.getRotationMatrix2D(image_center, 1, 1.0)
            imgs[i_tmp +   step] = cv.warpAffine(imgs[i_tmp], rot_mat, imgs[i_tmp].shape[1::-1], flags=cv.INTER_LINEAR, borderValue=0.78)
            rot_mat = cv.getRotationMatrix2D(image_center,-1, 1.0)
            imgs[i_tmp + 2*step] = cv.warpAffine(imgs[i_tmp], rot_mat, imgs[i_tmp].shape[1::-1], flags=cv.INTER_LINEAR, borderValue=0.78)
            rot_mat = cv.getRotationMatrix2D(image_center, 2, 1.0)
            imgs[i_tmp + 3*step] = cv.warpAffine(imgs[i_tmp], rot_mat, imgs[i_tmp].shape[1::-1], flags=cv.INTER_LINEAR, borderValue=0.78)
            rot_mat = cv.getRotationMatrix2D(image_center,-2, 1.0)
            imgs[i_tmp + 4*step] = cv.warpAffine(imgs[i_tmp], rot_mat, imgs[i_tmp].shape[1::-1], flags=cv.INTER_LINEAR, borderValue=0.78)
            rot_mat = cv.getRotationMatrix2D(image_center, 3, 1.0)
            imgs[i_tmp + 5*step] = cv.warpAffine(imgs[i_tmp], rot_mat, imgs[i_tmp].shape[1::-1], flags=cv.INTER_LINEAR, borderValue=0.78)
            rot_mat = cv.getRotationMatrix2D(image_center,-3, 1.0)
            imgs[i_tmp + 6*step] = cv.warpAffine(imgs[i_tmp], rot_mat, imgs[i_tmp].shape[1::-1], flags=cv.INTER_LINEAR, borderValue=0.78)
        end_size = int(end_size/7)

    if("Bright"  in aug_types):
        step = int(end_size/3)
        for i in range(0, orig_size*int(end_size/3)):
            i_tmp = end_size*i
            imgs[i_tmp + 1*step] = np.clip(imgs[i_tmp] + 0.1, 0, 1)
            imgs[i_tmp + 2*step] = np.clip(imgs[i_tmp] - 0.1, 0, 1)
        end_size = int(end_size/3)

    ofs = 12
    plt.subplot(4,4,1)
    plt.imshow(imgs[0 + ofs], cmap='grey')
    plt.subplot(4,4,2)
    plt.imshow(imgs[1 + ofs], cmap='grey')
    plt.subplot(4,4,3)
    plt.imshow(imgs[2 + ofs], cmap='grey')
    plt.subplot(4,4,4)
    plt.imshow(imgs[3 + ofs], cmap='grey')
    plt.subplot(4,4,5)
    plt.imshow(imgs[4 + ofs], cmap='grey')
    plt.subplot(4,4,6)
    plt.imshow(imgs[5 + ofs], cmap='grey')
    plt.subplot(4,4,7)
    plt.imshow(imgs[6 + ofs], cmap='grey')
    plt.subplot(4,4,8)
    plt.imshow(imgs[7 + ofs], cmap='grey')
    plt.subplot(4,4,9)
    plt.imshow(imgs[8 + ofs], cmap='grey')
    plt.subplot(4,4,10)
    plt.imshow(imgs[9 + ofs], cmap='grey')
    plt.subplot(4,4,11)
    plt.imshow(imgs[10 + ofs], cmap='grey')
    plt.subplot(4,4,12)
    plt.imshow(imgs[11 + ofs], cmap='grey')
    plt.subplot(4,4,13)
    plt.imshow(imgs[12 + ofs], cmap='grey')
    plt.subplot(4,4,14)
    plt.imshow(imgs[13 + ofs], cmap='grey')
    plt.subplot(4,4,15)
    plt.imshow(imgs[14 + ofs], cmap='grey')
    plt.subplot(4,4,16)
    plt.imshow(imgs[15 + ofs], cmap='grey')
    plt.show()

    return imgs, probs, types
    
@timefunc
def make_3_channel(imgs):
    imgs_3c = np.zeros([imgs.shape[0], imgs.shape[1], imgs.shape[2], 3])
    for i in range(0,imgs.shape[0]):
        imgs_3c[i] = cv.cvtColor(imgs[i], cv.COLOR_GRAY2BGR)
    return imgs_3c

@timefunc
def split_t_t_data(imgs, probs, types, out_types):
    if(out_types == "All" or out_types == "Mono"):
        train_m_imgs, test_m_imgs, train_m_probs, test_m_probs = \
            train_test_split(imgs[types == "mono"], probs[types == "mono"], test_size=0.25, random_state=0, shuffle=False)
        if(out_types == "Mono"):           
            return  train_m_imgs, train_m_probs, np.full([train_m_probs.shape[0],4], "poly"), test_m_imgs, test_m_probs, np.full([test_m_probs.shape[0],4], "poly")

    if(out_types == "All" or out_types == "Poly"):
        train_p_imgs, test_p_imgs, train_p_probs, test_p_probs = \
                train_test_split(imgs[types == "poly"], probs[types == "poly"], test_size=0.25, random_state=0, shuffle=False)
        if(out_types == "Poly"):
            return  train_p_imgs, train_p_probs, np.full([train_p_probs.shape[0],4], "poly"), test_p_imgs, test_p_probs, np.full([test_p_probs.shape[0],4], "poly")

    train_imgs = np.concatenate((train_m_imgs, train_p_imgs), axis=0) 
    del train_m_imgs, train_p_imgs
    train_probs = np.concatenate((train_m_probs, train_p_probs), axis=0)
    train_types = np.concatenate((np.full(train_m_probs.shape[0], "mono"), np.full(train_p_probs.shape[0], "poly")), axis=0)
    del train_m_probs, train_p_probs
    test_imgs  = np.concatenate((test_m_imgs,  test_p_imgs),  axis=0) 
    del test_m_imgs, test_p_imgs
    test_probs  = np.concatenate((test_m_probs,  test_p_probs),  axis=0)
    test_types = np.concatenate((np.full(test_m_probs.shape[0], "mono"), np.full(test_p_probs.shape[0], "poly")), axis=0)
    del test_m_probs, test_p_probs
    return train_imgs, train_probs, train_types, test_imgs, test_probs, test_types

@timefunc
def shuffle_set(train_imgs, train_probs, train_types, test_imgs, test_probs, test_types):
    rand_seed1 = np.random.randint(1, 2147483647)
    rand_seed2 = np.random.randint(1, 2147483647)
    np.random.seed(rand_seed1)
    np.random.shuffle(train_imgs)
    np.random.seed(rand_seed1)
    np.random.shuffle(train_probs)
    np.random.seed(rand_seed1)
    np.random.shuffle(train_types)
    np.random.seed(rand_seed2)
    np.random.shuffle(test_imgs)
    np.random.seed(rand_seed2)
    np.random.shuffle(test_probs)
    np.random.seed(rand_seed2)
    np.random.shuffle(train_types)

    return train_imgs, train_probs, train_types, test_imgs, test_probs, test_types


#load_and_preprocess_dataset()