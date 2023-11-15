from Dataloader import load_and_preprocess_dataset
import cnn as cnn
import numpy as np
import matplotlib.pyplot as plt

highest_accuracy_i = 0
highest_accuracy = 0

train_imgs, train_probs, train_types, test_imgs, test_probs, test_types = load_and_preprocess_dataset(augment="All", out_types="Mono", aug_types=["Flip", "Rot"], channels=3, balance_probs=0)

print("Train probs count: ", np.unique(train_probs, return_counts = True))
print("Test probs count: ", np.unique(test_probs, return_counts = True))

vgg19 = cnn.initialize_model("vgg19")

train_probs = cnn.onehot_encode(train_probs)
test_probs = cnn.onehot_encode(test_probs)

history = cnn.train_model(vgg19, train_imgs, train_probs, epochs = 100, filename="vgg19-base")

cnn.plot_loss(history)
cnn.plot_accuracy(history)

cnn.save_history(history, "vgg19-mono-base")

# history = cnn.finetune_model(vgg19, train_imgs, train_probs, path = "models/vgg19-mono-ft", epochs = 10)
# cnn.plot_loss(history)
# cnn.plot_accuracy(history)

score = cnn.evaluate_metrics(vgg19, test_imgs, test_probs)

cnn.predict_metrics(vgg19, test_imgs, test_probs)