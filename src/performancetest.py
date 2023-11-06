import numpy as np
import cv2 as cv
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib as plt

from elpvdata import ELPVData

class PerformanceTest:
    # NOTE: The use of 'classification_method' as a function reference is VERY sketchy
    # Potential solution is to hard-code the performance testing PER classification method once they have all been written
    def __init__(self, classification_method):
        self.classification_method = classification_method

        dataobj = ELPVData()
        self.training_data = dataobj.get_training_data()
        self.testing_data = dataobj.get_testing_data()

        training_mono_indices = [i for i, label in enumerate(self.training_data['labels']) if label == 'monocrystalline']
        training_poly_indices = [i for i, label in enumerate(self.training_data['labels']) if label == 'polycrystalline']
        self.training_mono_images = [self.training_data['images'][i] for i in training_mono_indices]
        self.training_poly_images = [self.training_data['images'][i] for i in training_poly_indices]

        testing_mono_indices = [i for i, label in enumerate(self.testing_data['labels']) if label == 'monocrystalline']
        testing_poly_indices = [i for i, label in enumerate(self.testing_data['labels']) if label == 'polycrystalline']
        self.testing_mono_images = [self.testing_data['images'][i] for i in testing_mono_indices]
        self.testing_poly_images = [self.testing_data['images'][i] for i in testing_poly_indices]

    def test_performance_all(self):
        predicted_labels = self.classification_method(self.training_data['images'], self.testing_data['images'])

        confusion_matrix_result = confusion_matrix(self.testing_data['labels'], predicted_labels)
        accuracy = accuracy_score(self.testing_data['labels'], predicted_labels)
        f1 = f1_score(self.testing_data['labels'], predicted_labels)

        return {
            'confusion_matrix': confusion_matrix_result,
            'accuracy': accuracy,
            'f1_score': f1
        }

    def test_performance_mono(self):
        predicted_labels = self.classification_method(self.training_mono_images, self.testing_mono_images)

        confusion_matrix_result = confusion_matrix(self.testing_data['labels'], predicted_labels)
        accuracy = accuracy_score(self.testing_data['labels'], predicted_labels)
        f1 = f1_score(self.testing_data['labels'], predicted_labels)

        return {
            'confusion_matrix': confusion_matrix_result,
            'accuracy': accuracy,
            'f1_score': f1
        }

    def test_performance_poly(self):
        predicted_labels = self.classification_method(self.training_poly_images, self.testing_poly_images)

        confusion_matrix_result = confusion_matrix(self.testing_data['labels'], predicted_labels)
        accuracy = accuracy_score(self.testing_data['labels'], predicted_labels)
        f1 = f1_score(self.testing_data['labels'], predicted_labels)

        return {
            'confusion_matrix': confusion_matrix_result,
            'accuracy': accuracy,
            'f1_score': f1
        }

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], fmt), horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
