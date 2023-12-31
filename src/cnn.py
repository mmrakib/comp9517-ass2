import numpy as np

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

import matplotlib.pyplot as plt
import pickle

def initialize_model(version = "vgg19"):
    if version == "type":
        type_model = keras.models.Sequential()
        type_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))
        type_model.add(keras.layers.MaxPooling2D((2, 2)))
        type_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        type_model.add(keras.layers.MaxPooling2D((2, 2)))
        type_model.add(keras.layers.Flatten())
        type_model.add(keras.layers.Dense(64, activation='relu'))
        type_model.add(keras.layers.Dense(2, activation = "softmax"))

        return type_model
 
    elif version == "vgg19":
        vgg19_base = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(240, 280, 3))
        vgg19_base.trainable = False

        vgg19_model = keras.models.Sequential([
        vgg19_base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Flatten(input_shape=vgg19_base.output_shape[1:]),
        keras.layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(2048, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(4, activation='softmax')
        ])

        return vgg19_model
    
def onehot_encode(y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = keras.utils.to_categorical(y)

    return y

#for initial dense layer training
def train_model(model, X_train, y_train, filename = None, optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08), batch_size = 16, epochs = 100, validation_split = 0.2):
       
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 12)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs = epochs, validation_split = validation_split, batch_size = batch_size, callbacks = [es])
 
    if filename != None:
        model.save("../models/" + filename + ".keras")
    return history

def finetune_model(model, X_train, y_train, filename = None, optimizer = keras.optimizers.SGD(learning_rate = 0.0005, momentum=0.9), batch_size = 16, epochs = 100, validation_split = 0.2, iterations = 1, unfreeze_loop = 2):

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10, min_delta=0)

    for i in range(iterations):
        for layer in model.layers[-(unfreeze_loop * (i + 1)):]:
            layer.trainable = True

        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs = epochs, validation_split = validation_split, batch_size = batch_size, callbacks = [es])
    
    if filename != None:
        model.save("../models/" + filename + ".keras")
    return history

def save_history(history, filename):
    with open('../histories/' + filename, 'wb') as file:
        pickle.dump(history.history, file)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0.5, 5])
    plt.legend(loc='lower right')
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def evaluate_metrics(model, X_test, y_test, filename=None):
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    if filename:
        with open('../outputs/' + filename + '_evaluate', 'wb') as f:
            pickle.dump(score, f)

    return score

def predict_metrics(model, X_test, y_test, filename=None):
    predict_probs = model.predict(X_test)
    predict_labels = np.argmax(predict_probs, axis = -1)
    y_test = np.argmax(y_test, axis = -1)

    print(classification_report(predict_labels, y_test))
    cf_matrix = confusion_matrix(y_test, predict_labels)
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=["0%", "33%", "66%", "100%"])
    disp.plot(xticks_rotation=45, cmap="viridis")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    print('F1 score: ', f1_score(predict_labels, y_test, average="weighted"))
    
    if filename:
        with open('../outputs/' + filename + '_cr', 'wb') as f:
            pickle.dump(classification_report(predict_labels, y_test), f)

    return predict_labels, y_test