import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import tensorflow as tf

def show_kernel(kernel, label=True, digits=None, text_size=28):
    # Format kernel
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)

    # Plot kernel
    cmap = plt.get_cmap('Blues_r')
    plt.imshow(kernel, cmap=cmap)
    rows, cols = kernel.shape
    thresh = (kernel.max()+kernel.min())/2
    # Optionally, add value labels
    if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            plt.text(j, i, val,
                     color=color, size=text_size,
                     horizontalalignment='center', verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


""" 
    FILTER WITH CONVOLUTION LAYER 
    - A convolutional layer carries out the filtering step.
     You might define a convolutional layer in a Keras 
     model something like this:
"""
from tensorflow import keras
from tensorflow.keras import layers

# With the filters parameter, you tell the convolutional
# layer how many feature maps you want it to create as output.
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    # More layers follow
])

"""
    DETECT WITH RELU
    - After filtering, the feature maps pass through the 
    activation function. A neuron with a rectifier attached
     is called a rectified linear unit. For that reason, 
     we might also call the rectifier function the ReLU 
     activation or even the ReLU function.
"""
# The ReLU activation can be defined in its own
# Activation layer, but most often you'll just
# include it as the activation function of Conv2D.
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3, activation='relu')
    # More layers follow
])

""" EXAMPLE - APPLY CONVOLUTION AND RELU """
# We'll do the extraction ourselves in this example to
# understand better what convolutional networks are
# doing "behind the scenes".

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

image_path = 'car_feature.jpeg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
# plt.show()

# For the filtering step, we'll define a kernel and then
# apply it with the convolution. The kernel in this case
# is an "edge detection" kernel. You can define it with
# tf.constant just like you'd define an array in Numpy
# with np.array. This creates a tensor of the sort TensorFlow uses.

kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])
plt.figure(figsize=(3, 3))
show_kernel(kernel)


# TensorFlow includes many common operations performed by
# neural networks in its tf.nn module. The two that we'll
# use are conv2d and relu. These are simply function
# versions of Keras layers.

# This block of code below does some reformatting to make
# things compatible with TensorFlow. The details aren't
# important for this example.

# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

# CONVOLUTION FUNCTION
# Now let's apply our kernel and see what happens.
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    # we'll talk about these two in lesson 4!
    strides=1,
    padding='SAME',
)
plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
# plt.show()

# RELU FUNCTION
# Next is the detection step with the ReLU function.
# This function is much simpler than the convolution,
# as it doesn't have any parameters to set.
image_detect = tf.nn.relu(image_filter)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.show()
plt.show(block=True)

