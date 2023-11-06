from tensorflow.keras import layers


def train_model(model, X_train, y_train, optimizer="adam", batch_size = 16, epochs = 100, validation_split = 0.25):
