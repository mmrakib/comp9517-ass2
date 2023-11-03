import elpv_reader
import sklearn.model_selection
import random

class ELPVData:
    def __init__(test_size):
        images, probs, types = elpv_reader.load_dataset()

        self.all_images = images
        self.all_probs = probs
        self.all_labels = types

        x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(images, types, test_size, random.randint(0, 100))

        self.training_images = x_train
        self.training_labels = y_train
        self.testing_images = x_test
        self.testing_labels = y_test

    def get_training_data():
        return {data: self.training_images, labels: self.training_labels}

    def get_training_labels():
        return {data: self.testing_images, labels: self.testing_labels}
