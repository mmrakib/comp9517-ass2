import elpv_reader
import sklearn.model_selection
import random

class ELPVData:
    _instance = None

    # Singleton, so we don't reload data
    def __new__(cls):
        if (cls.instance is None):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

    def __init__():
        images, probs, types = elpv_reader.load_dataset()
        x_train, x_test, y_train, y_test, probs_train, probs_test = sklearn.model_selection.train_test_split(
            images, types, probs, test_size=0.25, random_state=random.randint(0, 100))

        self.images = images
        self.probs = probs
        self.labels = types

        self.training_images = x_train
        self.training_labels = y_train
        self.training_probs = probs_train

        self.testing_images = x_test
        self.testing_labels = y_test
        self.testing_probs = probs_test

    def get_all_data(self):
        return {
            "images": self.images,
            "probs": self.probs,
            "labels": self.labels
        }

    def get_training_data(self):
        return {
            "images": self.training_images,
            "labels": self.training_labels,
            "probs": self.training_probs
        }

    def get_testing_data(self):
        return {
            "images": self.testing_images,
            "labels": self.testing_labels,
            "probs": self.testing_probs
        }
