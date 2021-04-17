"""
In this lesson we're going to see how we can
build neural networks capable of learning the
complex kinds of relationships deep neural
nets are famous for.
"""

# Stacking Dense Layers
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer
    layers.Dense(units=1),
])


""" EXERCISE SECTION """
concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
concrete.head()

# 1. INPUT SHAPE
# The target for this task is the column 'CompressiveStrength'.
# The remaining columns are the features we'll use as inputs.
# What would be the input shape for this dataset?
print("---------- Number 1 ----------")
input_shape = [concrete.shape[1]-1]
print(input_shape)

# 2. DEFINE A MODEL WITH HIDDEN LAYERS
# Now create a model with three hidden layers, each having
# 512 units and the ReLU activation. Be sure to include an
# output layer of one unit and no activation, and also input_
# shape as an argument to the first layer.
print("---------- Number 2 ----------")
model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1)
])

# 3. ACTIVATION LAYERS
# Rewrite the following model so that each activation is
# in its own Activation layer.
print("---------- Number 3 ----------")
model = keras.Sequential([
    layers.Dense(units=32, input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(units=32),
    layers.Activation('relu'),
    layers.Dense(units=1),
])





