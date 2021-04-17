""" CONTINUATION OF EXERCISE 2 - WITH MAXIMUM POOLING """

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Read image
image_path = 'car_illus.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

# Embossing kernel
kernel = tf.constant([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
])

# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='VALID',
)

image_detect = tf.nn.relu(image_filter)

# Show what we have so far
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.title('Input')
plt.subplot(132)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Filter')
plt.subplot(133)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Detect')
# plt.show()

""" #1 APPLYING POOLING CONDENSE """
# YOUR CODE HERE
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)

# COMPARES IMAGE AFTER RELU AND AFTER MAX POOLING
plt.figure(figsize=(8, 6))
plt.subplot(121)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title("Detect (ReLU)")
plt.subplot(122)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title("Condense (MaxPool)")


""" #2 EXPLORE INVARIANCE """
#  In the tutorial, we talked about how maximum pooling
#  creates translation invariance over small distances.
#  This means that we would expect small shifts to disappear
#  after repeated maximum pooling. If you run the cell
#  multiple times, you can see the resulting image is
#  always the same; the pooling operation destroys those
#  small translations.

""" GLOBAL AVERAGE POOLING """
# - This is global average pooling. A GlobalAvgPool2D
# layer is often used as an alternative to some or all
# of the hidden Dense layers in the head of the network,
# like so:

# model = keras.Sequential([
#     pretrained_base,
#     layers.GlobalAvgPool2D(),
#     layers.Dense(1, activation='sigmoid'),
# ])

# Load VGG16
pretrained_base = tf.keras.models.load_model('vgg16-pretrained-base',)

model = keras.Sequential([
    pretrained_base,
    # Attach a global average pooling layer after the base
    layers.GlobalAvgPool2D(),
])

# Load dataset
ds = image_dataset_from_directory(
    'train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=1,
    shuffle=True,
)

ds_iter = iter(ds)
car = next(ds_iter)

car_tf = tf.image.resize(car[0], size=[128, 128])
car_features = model(car_tf)
car_features = tf.reshape(car_features, shape=(16, 32))
label = int(tf.squeeze(car[1]).numpy())

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(tf.squeeze(car[0]))
plt.axis('off')
plt.title(["Car", "Truck"][label])
plt.subplot(122)
plt.imshow(car_features)
plt.title('Pooled Feature Maps')
plt.axis('off')

plt.show()

""" #3 UNDERSTAND THE POOLED FEATURES """
# The VGG16 base produces 512 feature maps. We can think of each feature map
# as representing some high-level visual feature in the original image -- maybe
# a wheel or window. Pooling a map gives us a single number, which we could
# think of as a score for that feature: large if the feature is present, small
# if it is absent. Cars tend to score high with one set of features, and Trucks
# score high with another. Now, instead of trying to map raw features to classes,
# the head only has to work with these scores that GlobalAvgPool2D produced, a
# much easier problem for it to solve.

# Global average pooling is often used in modern convnets. One big advantage
# is that it greatly reduces the number of parameters in a model, while still
# telling you if some feature was present in an image or not -- which for
# classification is usually all that matters. If you're creating a
# convolutional classifier it's worth trying out!
