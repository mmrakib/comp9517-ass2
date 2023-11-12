from src.Dataloader import load_and_preprocess_dataset
import src.cnn as cnn


from elpv.utils.elpv_reader import load_dataset
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

# imgs, probs, types = load_dataset()
# imgs = imgs.astype("float32") / 255

# imgs_3c = np.zeros([imgs.shape[0], imgs.shape[1], imgs.shape[2], 3])
# for i in range(0,imgs.shape[0]):
#     imgs_3c[i] = cv.cvtColor(imgs[i], cv.COLOR_GRAY2BGR)

# train_imgs, test_imgs, train_probs, test_probs = train_test_split(imgs_3c[types == "mono"], probs[types == "mono"], test_size=0.25, random_state=0, shuffle=True, stratify=probs[types == "mono"])

train_imgs, train_probs, train_types, test_imgs, test_probs, test_types = load_and_preprocess_dataset(augment="All", out_types="Mono", aug_types = ["Flip", "Rot"], channels=3)

vgg19_base = cnn.initialize_model("vgg19")

train_probs = cnn.onehot_encode(train_probs)
test_probs = cnn.onehot_encode(test_probs)


history = cnn.train_model(vgg19_base, train_imgs, train_probs, path = "", epochs = 10)
cnn.plot_loss(history)
cnn.plot_accuracy(history)

cnn.evaluate_metrics(vgg19_base, test_imgs, test_probs)



