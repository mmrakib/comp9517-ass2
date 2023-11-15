from Dataloader import load_and_preprocess_dataset
import cnn as cnn
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# types = ["Mono", "Poly"]

# for type in types:

#     train_imgs, train_probs, train_types, test_imgs, test_probs, test_types = load_and_preprocess_dataset(augment="All", out_types=type, aug_types=["Flip"], channels=3, balance_probs=0)

#     print("Train probs count: ", np.unique(train_probs, return_counts = True))
#     print("Test probs count: ", np.unique(test_probs, return_counts = True))

#     train_probs = cnn.onehot_encode(train_probs)
#     test_probs = cnn.onehot_encode(test_probs)


#     vgg19 = cnn.initialize_model("vgg19")
#     history = cnn.train_model(vgg19, train_imgs, train_probs, epochs = 100, filename=f'vgg19-{type}-base')

#     cnn.plot_loss(history)
#     cnn.plot_accuracy(history)

#     cnn.save_history(history, "vgg19-mono-base")

#     history = cnn.finetune_model(vgg19, train_imgs, train_probs, iterations=3, filename = f'vgg19-{type}-finetuned', epochs = 100)
#     cnn.plot_loss(history)
#     cnn.plot_accuracy(history)

#     # vgg19 = tf.keras.models.load_model('../models/vgg19-base')

#     score = cnn.evaluate_metrics(vgg19, test_imgs, test_probs, filename = f'vgg19-{type}-finetuned')

    # cnn.predict_metrics(vgg19, test_imgs, test_probs, filename= f'vgg19-{type}-finetuned')

train_imgs, train_probs, train_types, test_imgs, test_probs, test_types = load_and_preprocess_dataset(augment="All", out_types="Mono", aug_types=["Flip"], channels=3, balance_probs=0)

train_probs = cnn.onehot_encode(train_probs)
test_probs = cnn.onehot_encode(test_probs)

vgg19 = tf.keras.models.load_model('../models/vgg19-mono-ft.keras')

score = cnn.evaluate_metrics(vgg19, test_imgs, test_probs, filename = f'vgg19-mono-finetuned')

cnn.predict_metrics(vgg19, test_imgs, test_probs, filename= f'vgg19-mono-finetuned')