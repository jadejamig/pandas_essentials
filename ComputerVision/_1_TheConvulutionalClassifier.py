""" Convolutional Classifier - includes convolutional base and dense head """
# Convolutional Base - which features to extract from the image
# Dense Head - which class goes with what features

# STEP 1 - LOAD DATA

# Imports
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(31415)

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    'train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    'valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# STEP 2 - DEFINE PRETRAINED BASE
pretrained_base = tf.keras.models.load_model('vgg16-pretrained-base',)
pretrained_base.trainable = False

# STEP 3 - ATTACH HEAD
# Next, we attach the classifier head. For this example,
# we'll use a layer of hidden units (the first Dense layer)
# followed by a layer to transform the outputs to a probability
# score for class 1, Truck. The Flatten layer transforms the
# two dimensional outputs of the base into the one dimensional
# inputs needed by the head.
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# STEP 4 - TRAIN
# Finally, let's train the model. Since this is a two-class
# problem, we'll use the binary versions of crossentropy and
# accuracy. The adam optimizer generally performs well, so
# we'll choose it as well.
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    # verbose=0,
)
print("lol")
# When training a neural network, it's always a good idea to
# examine the loss and metric plots. The history object
# contains this information in a dictionary history.history.
# We can use Pandas to convert this dictionary to a dataframe
# and plot it with a built-in method.

import pandas as pd

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show(block=True)


# SOBRANG BAGAL MAG TRAIN NG MODEL DI KO MATAPOS HANGGANG DULO
# HAHAHAHAHAHAHAHAHAHAHAHA