import pandas as pd
import matplotlib.pyplot as plt

red_wine = pd.read_csv('red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
# remove instances of df_train then store to df_valid
df_valid = red_wine.drop(df_train.index)
print(df_train.head(4))

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


from tensorflow import keras
from tensorflow.keras import layers

# Eleven columns means eleven inputs.
# We've chosen a three-layer network with over 1500 neurons.
# This network should be capable of learning fairly complex
# relationships in the data.
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

# After defining the model, we compile in the optimizer and loss function.
model.compile(
    optimizer='adam',
    loss='mae',
)

# We've told Keras to feed the optimizer 256 rows of the
# training data at a time (the batch_size) and to do that
# 10 times all the way through the dataset (the epochs).

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)

# Often, a better way to view the loss though is to plot it.
# The fit method in fact keeps a record of the loss produced
# during training in a History object. We'll convert the data
# to a Pandas dataframe, which makes the plotting easy.

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot()
plt.show(block=True)


""" EXERCISE SECTION """



