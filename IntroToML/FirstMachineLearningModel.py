import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv('melb_data.csv')

""" SELECTING DATA FOR MODELLING """

print("---------- List of columns of a Dataframe ----------")
print(melbourne_data.columns)

# dropna drops missing values (think of na as "not available")
print("---------- Drop na values in the Dataframe ----------")
melbourne_data = melbourne_data.dropna(axis=0)
print(melbourne_data)


""" SELECTING THE PREDICTION TARGET """

# The prediction target is a column in a dataframe and is also called "y"
print("---------- The Prediction Target ----------")
y = melbourne_data.Price
print(y)


""" CHOOSING FEATURES """

# FEATURES are the columns that will be used to make predictions
# sometimes you will use all the columns except the target as features
print("---------- The Features ----------")
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]
print(x)


""" BUILDING YOUR MODEL """

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(x, y)

print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(melbourne_model.predict(x.head()))