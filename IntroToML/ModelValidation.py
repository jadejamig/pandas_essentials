import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load data
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]


# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)

# MAKE PREDICTION
print("---------- Prediction ----------")
predicted_home_prices = melbourne_model.predict(X)
print(predicted_home_prices)
# MEAN ABSOLUTE ERROR
mae = mean_absolute_error(y, predicted_home_prices)
print("---------- Mean Absolute Error ----------")
print(mae)


""" Validating the data by splitting the it
    using the other half to build the model
    and the other half to test the model """

# TRAIN_TEST_SPLIT function - used to break the data int o two pieces
# some data will be used to fit the model and the other will be used
# as validation data to calculate the MEAN_ABSOLUTE_ERROR


# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
mae = mean_absolute_error(val_y, val_predictions)
print("---------- Mean Absolute Error with TRAIN_TEST_SPLIT----------")
print(mae)


