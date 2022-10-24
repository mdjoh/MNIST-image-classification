"""
Image classification using a feedforward artificial neural network
Classify handwritten digits from the MNIST dataset
"""

# Import modules
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Data handling
def load_data():
    # Import MNIST dataset from openml
    dataset = fetch_openml('mnist_784', version=1, data_home=None)

    # Data preparation
    raw_X = dataset['data']
    raw_Y = dataset['target']
    return raw_X, raw_Y

raw_X, raw_Y = load_data()

# Clean data by scaling image pixel data between 0 and 1 and converting true labels to categorical classes
def clean_data(raw_X, raw_Y):
    cleaned_X = raw_X / 255.0

    cleaned_Y = keras.utils.to_categorical(raw_Y, num_classes=10, dtype="float32")

    return cleaned_X, cleaned_Y

cleaned_X, cleaned_Y = clean_data(raw_X, raw_Y)

# Split data into train, validation, and test sets by following a 50-20-30 ratio
def split_data(cleaned_X, cleaned_Y):

    # Training set
    X_train, X_temp, Y_train, Y_temp = train_test_split(cleaned_X, cleaned_Y, test_size=0.5, random_state=42)

    # From X_temp, Y_temp, get validation and test sets
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.6, random_state=42)

    return X_val, X_test, X_train, Y_val, Y_test, Y_train

X_val, X_test, X_train, Y_val, Y_test, Y_train = split_data(cleaned_X, cleaned_Y)

# Visualize data
def visualize_data(X_train, Y_train, n_samples):
    X_train_sample = X_train.to_numpy()[:n_samples]

    Y_train_sample = Y_train[:n_samples]

    plt.figure(figsize=(10,10))

    for i in range(n_samples):
      plt.subplot(2, n_samples/2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(X_train_sample[i].reshape(28,28), cmap=plt.cm.binary)
      plt.xlabel(f"Digit: {Y_train_sample[i].argmax()}")

    plt.show()

visualize_data(X_train, Y_train, 10)

# Build model with 2 hidden layers with ReLu activation function and 20% dropout rate
# softmax activation function in the final layer
def build_model():

    model = Sequential()
    model.add(keras.layers.Flatten(input_shape=(784,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    return model

model = build_model()

# Compile and train model
def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, Y_train, X_val, Y_val):
    history = model.fit(x=X_train, y=Y_train, batch_size=128, epochs=12, verbose=2, validation_data=(X_val, Y_val), shuffle=True)

    return model, history

model = compile_model(model)
model, history = train_model(model, X_train, Y_train, X_val, Y_val)

# Model evaluation
def eval_model(model, X_test, Y_test):
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)

    return test_loss, test_accuracy

test_loss, test_accuracy = eval_model(model, X_test, Y_test)

# Model prediction
predictions = model.predict(X_test)

# Visualize predictions
def visualize_predictions(X_test, Y_test, predictions, n_samples):
    X_test_sample = X_test.to_numpy()[:n_samples]

    Y_test_sample = Y_test[:n_samples]

    plt.figure(figsize=(10,10))

    for i in range(len(X_test_sample)):
      plt.subplot(2, n_samples/2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(X_test_sample[i].reshape(28,28), cmap=plt.cm.binary)
      plt.xlabel(f"Predicted Digit: {predictions[i].argmax()}\nActual Digit: {Y_test_sample[i].argmax()}")

    plt.show()

visualize_predictions(X_test, Y_test, predictions, 10)
