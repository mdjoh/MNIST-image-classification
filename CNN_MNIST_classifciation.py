"""
Image classification using a convolutional neural network
Classify handwritten digits from the MNIST dataset
"""

# Import modules
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import tensorflow.keras as keras

# Data handling
def load_data():
    # Import MNIST dataset from openml
    dataset = fetch_openml('mnist_784', version=1, data_home=None)

    # Data preparation
    raw_X = dataset['data']
    raw_Y = dataset['target']
    return raw_X, raw_Y

raw_X, raw_Y = load_data()

# Clean data
def clean_data(raw_X, raw_Y):

    # X transformation: scale image pixel data between -1 and 1 and reshape X (70,000 images) to be 28x28
    scaled_X = raw_X.to_numpy() / 127.5 - 1.0
    cleaned_X = scaled_X.reshape(70000, 28, 28, 1)

    # Y transformation: convert true labels to categorical classes
    cleaned_Y = keras.utils.to_categorical(raw_Y, num_classes=10, dtype="float32")

    return cleaned_X, cleaned_Y

cleaned_X, cleaned_Y = clean_data(raw_X, raw_Y)

# Split data into train, validation, and test sets by following a 50-20-30 ratio
def split_data(cleaned_X, cleaned_Y):

    # Training set
    X_train, X_test, Y_train, Y_test = train_test_split(cleaned_X, cleaned_Y, test_size=0.5, random_state=42)

    # From X_test, Y_test, get validation and test sets
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.6, random_state=42)

    return X_val, X_test, X_train, Y_val, Y_test, Y_train

X_val, X_test, X_train, Y_val, Y_test, Y_train = split_data(cleaned_X, cleaned_Y)

# Build model
def build_model():

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=[28, 28, 1], padding='same'))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=1))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=10, activation='softmax'))

    return model

# Compile and train model
def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, Y_train, X_val, Y_val):
    history = model.fit(x=X_train, y=Y_train, batch_size=256, epochs=14, verbose=2, validation_data=(X_val, Y_val), shuffle=True)

    return model, history

# Evaluate model
def eval_model(model, X_test, Y_test):
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)

    return test_loss, test_accuracy

# Deploy model on data
model = build_model()
model = compile_model(model)
model, history = train_model(model, X_train, Y_train, X_val, Y_val)
test_loss, test_accuracy = eval_model(model, X_test, Y_test)
