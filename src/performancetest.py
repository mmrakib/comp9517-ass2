import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics         as sk_met


def display_results(y_true_m=[], y_true_p=[], y_predict_m=[], y_predict_p=[]):
    y_true    = np.concatenate((y_true_m,    y_true_p),    axis=0) 
    y_predict = np.concatenate((y_predict_m, y_predict_p), axis=0) 

    if y_true_m != [] and y_predict_m != []:
        print_scores(y_true_m, y_predict_m, "Mono Results    ")
    if y_true_p != [] and y_predict_p != []:
        print_scores(y_true_p, y_predict_p, "Poly Results    ")
        if y_true_m != [] and y_predict_m != []:
            print_scores(y_true,   y_predict, "Combined Results")
    
    display_conf_mat([y_true_m, y_true, y_true_p], [y_predict_m, y_predict, y_predict_p])

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
        " Precision:",  round(sk_met.precision_score(y_true, y_predict, average='macro', zero_division=np.nan),5),
        " Recall:",     round(sk_met.recall_score(   y_true, y_predict, average='macro'),5),
        " F1:",         round(sk_met.f1_score(       y_true, y_predict, average='macro'),5))
    
def display_conf_mat(y_true_lst, y_predict_lst):
    f, axes = plt.subplots(1, 3, figsize=(20, 5), sharey='row')
    prob_name_lst = ["0%", "33%", "66%", "100%"]
    plt_names = ["Mono", "Combined", "Poly"]
    for i in range(0,3):
        if y_true_lst[i] != [] and y_predict_lst[i] != []:
            cf_matrix = sk_met.confusion_matrix(y_true_lst[i], y_predict_lst[i])
        disp = sk_met.ConfusionMatrixDisplay(cf_matrix, display_labels=prob_name_lst)
        disp.plot(ax=axes[i], xticks_rotation=45)
        disp.ax_.set_title(plt_names[i])
        disp.ax_.set_xlabel('')
        if i!=0:
            disp.ax_.set_ylabel('')

    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()

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
    pt2.title.set_text("Accuracy")
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='lower right')
    plt.show()

