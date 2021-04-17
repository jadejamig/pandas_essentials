""" Wider - wider output """
# wider = keras.Sequential([
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1),
# ])
#
""" Deeper - mmore layers """
# deeper = keras.Sequential([
#     layers.Dense(16, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1),
# ])

import pandas as pd
from IPython.display import display

red_wine = pd.read_csv('red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

# Now let's increase the capacity of the network. We'll go for
# a fairly large network, but rely on the callback to halt the
# training once the validation loss shows signs of increasing.

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import matplotlib.pyplot as plt

""" ADDING EARLY STOPPING """
# EARLY STOPPING PREVENTS OVERFITTING
# These parameters say: "If there hasn't been at least an
# improvement of 0.001 in the validation loss over the previous
# 20 epochs, then stop the training and keep the best model you found.
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss='mae',
)

# After defining the callback, add it as an argument in fit
# (you can have several, so put it in a list). Choose a large
# number of epochs when using early stopping, more than you'll need.

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

history_df = pd.DataFrame(history.history)

history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
plt.show(block=True)