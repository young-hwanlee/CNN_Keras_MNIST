#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

## Import libraries and modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import utils
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

## Set up the seed for reproducibility.
seed=42
tf.reset_default_graph()
tf.set_random_seed(seed)
np.random.seed(seed)

## Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

## Reserve 10,000 samples for validation
X_train = X_train[:-10000]
y_train = y_train[:-10000]
X_val = X_train[-10000:]
y_val = y_train[-10000:]

## Preprocess class labels
Y_train = utils.to_categorical(y_train, 10)
Y_val = utils.to_categorical(y_val, 10)
Y_test = utils.to_categorical(y_test, 10)

## Define model architecture
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

## Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

## Fit model on training data
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    batch_size=32, nb_epoch=10, verbose=1)

## Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

#%%
## Check the results
# Plots the loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Plots the accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

## Predict using the test dataset
Y_pred = model.predict(X_test)
X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

fig, axis = plt.subplots(2, 5, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test__[i], cmap='binary')
    ax.set(title = f"Real Number is {Y_test[i].argmax()}\n"
                   f"Predict Number is {Y_pred[i].argmax()}")


