# from elpv.utils.elpv_reader import load_dataset
import numpy as np

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import cv2 as cv
import matplotlib.pyplot as plt

def initialize_model(version = "vgg19"):