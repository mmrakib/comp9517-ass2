import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics         as sk_met


def display_results(y_true_m, y_true_p, y_predict_m, y_predict_p):
    y_true    = np.concatenate((y_true_m,    y_true_p),    axis=0) 
    y_predict = np.concatenate((y_predict_m, y_predict_p), axis=0) 

    print_scores(y_true_m, y_predict_m, "Mono Results")
    print_scores(y_true_p, y_predict_p, "Poly Results")
    print_scores(y_true,   y_predict, "Combined Results")
    
    plt.title("Confusion Matrix")
    pt1 = plt.subplot(2,2,1)
    pt1.title.set_text("Mono")
    sk_met.ConfusionMatrixDisplay(sk_met.confusion_matrix(y_true_m, y_predict_m)).plot()
    pt2 = plt.subplot(2,2,2)
    pt2.title.set_text("Poly")
    sk_met.ConfusionMatrixDisplay(sk_met.confusion_matrix(y_true_p, y_predict_p)).plot()
    pt3 = plt.subplot(2,1,2)
    pt3.title.set_text("Combined")
    sk_met.ConfusionMatrixDisplay(sk_met.confusion_matrix(y_true, y_predict)).plot()
    plt.show()

def predict_results(prediction_func, model, test_imgs, test_probs, test_types):
    imgs_m = test_imgs[test_types == "mono"]
    imgs_p = test_imgs[test_types == "poly"]

    probs_m = test_probs[test_types == "mono"]
    probs_p = test_probs[test_types == "poly"]

    y_true_m, y_predict_m = prediction_func(model, imgs_m, probs_m)
    y_true_p, y_predict_p = prediction_func(model, imgs_p, probs_p)
    return y_true_m, y_true_p, y_predict_m, y_predict_p

def print_scores(y_true, y_predict, title):
    print(title, "|",
        " Accuracy:",   round(sk_met.accuracy_score( y_true, y_predict),5),
        " Precision:",  round(sk_met.precision_score(y_true, y_predict, average='macro'),5),
        " Recall:",     round(sk_met.recall_score(   y_true, y_predict, average='macro'),5),
        " F1:",         round(sk_met.f1_score(       y_true, y_predict, average='macro'),5))
    
def plot_train_data(history):
    #loss
    pt1 = plt.subplot(1,2,1)
    pt1.title.set_text("Loss")
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log',base=2) 
    plt.legend(loc='upper right')

    #accuracy
    pt2 = plt.subplot(1,2,2)
    pt2.title.set_text("Mono")
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='lower right')
    plt.show()

