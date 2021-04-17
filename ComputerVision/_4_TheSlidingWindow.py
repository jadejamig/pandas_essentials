
# The convolution and pooling operations share a common
# feature: they are both performed over a sliding window.
# With convolution, this "window" is given by the dimensions
# of the kernel, the parameter kernel_size. With pooling,
# it is the pooling window, given by pool_size.

# There are two additional parameters affecting both convolution
# and pooling layers -- these are the STRIDES of the window and
# whether to use PADDING at the image edges. The strides parameter
# says how far the window should move at each step, and the padding
# parameter describes how we handle the pixels at the edges of the input.

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  activation='relu'),
    layers.MaxPool2D(pool_size=2,
                     strides=1,
                     padding='same')
    # More layers follow
])

# STRIDES
# Whenever the stride in either direction is greater than 1, the sliding
# window will skip over some of the pixels in the input at each step.

# COMMON STRIDE VALUES
# FOR CONVOLUTIONAL LAYERS - mostly have a stride of (1, 1)
# FOR MAXIMUM POOLING LAYERS - always have stride values greater
# than 1 but not larger than the window itself like (2, 2) or (3, 3)

# Finally, note that when the value of the strides is the same number
# in both directions, you only need to set that number; for instance,
# instead of strides=(2, 2), you could use strides=2 for the parameter setting.

# PADDING
# padding='same'
# The trick here is to pad the input with 0's around its borders,
# using just enough 0's to make the size of the output the same as
# the size of the input. This can have the effect however of
# diluting the influence of pixels at the borders.

# padding='valid'
# When we set padding='valid', the convolution window will stay
# entirely inside the input. The drawback is that the output
# shrinks (loses pixels), and shrinks more for larger kernels.
# This will limit the number of layers the network can contain,
# especially when inputs are small in size.

""" EXAMPLE - EXPLORING SLIDING WINDOWS """
# To better understand the effect of the sliding window parameters,
# it can help to observe a feature extraction on a low-resolution
# image so that we can see the individual pixels. Let's just look
# at a simple circle.

# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# plt.rc('figure', autolayout=True)
# plt.rc('axes', labelweight='bold', labelsize='large',
#        titleweight='bold', titlesize=18, titlepad=10)
# plt.rc('image', cmap='magma')
#
# image = circle([64, 64], val=1.0, r_shrink=3)
# image = tf.reshape(image, [*image.shape, 1])
# # Bottom sobel
# kernel = tf.constant(
#     [[-1, -2, -1],
#      [0, 0, 0],
#      [1, 2, 1]],
# )
#
# show_kernel(kernel)