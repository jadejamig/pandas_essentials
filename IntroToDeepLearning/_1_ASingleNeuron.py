import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

""" LINEAR UNIT / SINGLE NEURON MODEL """
    # Formula: y = w * x + b
    # slope-intercept equation

# Create a network with 1 linear unit
# 3 input and 1 output
# model = keras.Sequential([
#     layers.Dense(units=1, input_shape=[3])
# ])

red_wine = pd.read_csv('red-wine.csv')
print(red_wine.head())

# (rows, columns)
print("---------- Number of Rows and Columns ----------")
print(red_wine.shape)


""" EXERCISE SECTION """

# 1. INPUT SHAPE
# The target is 'quality', and the remaining
# columns are the features. How would you set
# the input_shape parameter for a Keras model
# on this task?
print("---------- Number 1 ----------")
input_shape = [red_wine.shape[1]-1]
print(input_shape)

# 2. DEFINE A LINEAR MODEL
# Now define a linear model appropriate for
# this task. Pay attention to how many inputs
# and outputs the model should have.
print("---------- Number 2 ----------")
model = keras.Sequential([
    layers.Dense(units=1, input_shape=input_shape)
])

# 3. LOOK AT THE WEIGHTS (also called TENSORS)
# Internally, Keras represents the weights of a neural
# network with tensors. Tensors are basically TensorFlow's
# version of a Numpy array with a few differences that
# make them better suited to deep learning. One of the most
# important is that tensors are compatible with GPU and TPU)
# accelerators. TPUs, in fact, are designed specifically
# for tensor computations. A model's weights are kept in
# its weights attribute as a list of tensors
print("---------- Number 3 ----------")
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))


""" OPTIONAL """
import tensorflow as tf
import matplotlib.pyplot as plt

model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()