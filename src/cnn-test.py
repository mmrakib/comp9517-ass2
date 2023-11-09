from Dataloader import load_and_preprocess_dataset
import cnn

train_imgs, train_probs, train_types, test_imgs, test_probs, test_types = load_and_preprocess_dataset(augment="All", out_types="Mono", aug_types = ["Flip"], channels=3)

train_probs = cnn.onehot_encode(train_probs)
test_probs = cnn.onehot_encode(test_probs)

dumbass_model = cnn.initialize_model("dumbass")

history = cnn.train_model(dumbass_model, train_imgs, train_probs, path = "", epochs = 10)
cnn.plot_loss(history)
cnn.plot_accuracy(history)

cnn.evaluate_metrics(dumbass_model, test_imgs, test_probs)



