import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

""" APPLY TRANSFORMATIONS """
# The next few exercises walk through feature extraction
# just like the example in the tutorial. Run the following
# cell to load an image we'll use for the next few exercises.
image_path = 'car_illus.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

img = tf.squeeze(image).numpy()
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.axis('off')
# plt.show()

""" #1 DEFINE KERNEL """
# One thing to keep in mind is that the sum of the
# numbers in the kernel determines how bright the
# final image is. Generally, you should try to keep
# the sum of the numbers between 0 and 1

kernel = tf.constant([
       [-1, -2, -1],
       [0, 0, 0],
       [1, 2, 1],
])

# Now we'll do the first step of feature extraction,
# the filtering step. First run this cell to do some
# reformatting for TensorFlow.

# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

""" #2 APPLY CONVULUTION """
# Now we'll apply the kernel to the image by a convolution.
# The layer in Keras that does this is layers.Conv2D. What
# is the backend function in TensorFlow that performs the
# same operation?

# YOUR CODE HERE: Give the TensorFlow convolution function (without arguments)
conv_fn = tf.nn.conv2d

# Once you've got the correct answer, run this next cell
# to execute the convolution and see the result!
image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1,  # or (1, 1)
    padding='SAME',
)

plt.figure(figsize=(6, 6))
plt.imshow(
    # Reformat for plotting
    tf.squeeze(image_filter)
)
plt.axis('off')
# plt.show()


""" #3 APPLY RELU """
# Now detect the feature with the ReLU function. In Keras,
# you'll usually use this as the activation function in a
# Conv2D layer. What is the backend function in TensorFlow
# that does the same thing?

# YOUR CODE HERE: Give the TensorFlow ReLU function (without arguments)
relu_fn = tf.nn.relu

image_detect = relu_fn(image_filter)
plt.figure(figsize=(6, 6))
plt.imshow(
    # Reformat for plotting
    tf.squeeze(image_detect)
)
plt.axis('off')



""" MATHEMATICAL / COMPUTATIONAL PERSPECTIVE OF CONVOLUTION AND RELU """
# Let's start by defining a simple array to act as
# an image, and another array to act as the kernel.
# Run the following cell to see these arrays.

# Sympy is a python library for symbolic mathematics. It has a nice
# pretty printer for matrices, which is all we'll use it for.
import sympy
sympy.init_printing()


image = np.array([
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
])

kernel = np.array([
    [1, -1],
    [1, -1],
])

# USING SYMPY
print(sympy.Matrix(image))
print(sympy.Matrix(kernel))
# USING NORMAL NP STRUCTURE
print(image)
print(kernel)

# Reformat for Tensorflow
image = tf.cast(image, dtype=tf.float32)
image = tf.reshape(image, [1, *image.shape, 1])
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

plt.figure(figsize=(6, 6))
plt.title('original image')
plt.imshow(
    # Reformat for plotting
    tf.squeeze(image)
)
plt.axis('off')

plt.figure(figsize=(3, 3))
plt.title('Kernel')
plt.imshow(
    # Reformat for plotting
    tf.squeeze(kernel)
)
plt.axis('off')


# In the tutorial, we talked about how the pattern
# of positive numbers will tell you the kind of
# features the kernel will extract. This kernel has
# a vertical column of 1's, and so we would expect
# it to return features of vertical lines.

image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='VALID',
)
image_detect = tf.nn.relu(image_filter)

# The first matrix is the image after convolution,
# and the second is the image after ReLU.

# AFTER CONV2D IMAGE
plt.figure(figsize=(6, 6))
plt.title('image filter')
plt.imshow(
    # Reformat for plotting
    tf.squeeze(image_filter)
)
plt.axis('off')

# AFTER RELU IMAGE
plt.figure(figsize=(6, 6))
plt.title('image detect')
plt.imshow(
    # Reformat for plotting
    tf.squeeze(image_detect)
)
plt.axis('off')

plt.show()
plt.show(block=True)