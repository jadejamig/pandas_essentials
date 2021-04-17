""" ACCURACY - Accuracy is one of the many metrics in use for
               measuring success on a classification problem."""

""" CROSS ENTROPY - Cross-entropy is a sort of measure for the
                    distance from one probability distribution to another.
                    Use for classification problems. """

""" SIGMOID ACTIVATION - To covert the real-valued outputs 
                         produced by a dense layer into probabilities. """

import pandas as pd

ion = pd.read_csv('ion.csv', index_col=0)
print(ion.head())

df = ion.copy()
df['Class'] = df['Class'].map({'good': 0, 'bad': 1})

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
df_train.dropna(axis=1, inplace=True) # drop the empty feature in column 2
df_valid.dropna(axis=1, inplace=True)

X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)
y_train = df_train['Class']
y_valid = df_valid['Class']

# We'll define our model just like we did for the regression
# tasks, with one exception. In the final layer include a
# 'sigmoid' activation so that the model will produce class
# probabilities.

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Add the cross-entropy loss and accuracy metric to the model
# with its compile method. For two-class problems, be sure to
# use 'binary' versions. (Problems with more classes will be
# slightly different.) The Adam optimizer works great for
# classification too, so we'll stick with it.

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# The model in this particular problem can take quite a few
# epochs to complete training, so we'll include an early
# stopping callback for convenience.

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)

# We'll take a look at the learning curves as always, and
# also inspect the best values for the loss and accuracy we
# got on the validation set. (Remember that early stopping
# will restore the weights to those that got these values.)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(),
              history_df['val_binary_accuracy'].max()))
plt.show(block=True)