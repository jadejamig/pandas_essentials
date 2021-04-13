
# OVERFITTING
""" This is a phenomenon called overfitting where a model
    matches the training data almost perfectly, but does
    poorly in validation and other new data. On the flip side,
     if we make our tree very shallow, it doesn't divide up
     the houses into very distinct groups."""

# UNDERFITTING
""" At an extreme, if a tree divides houses into only 2 or 4, 
    each group still has a wide variety of houses. Resulting 
    predictions may be far off for most houses, even in the 
    training data (and it will be bad in validation too for 
    the same reason). When a model fails to capture important 
    distinctions and patterns in the data, so it performs 
    poorly even in training data, that is called underfitting."""


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


# FUNCTION TO GET MEAN ABSOLUTE ERROR
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


import pandas as pd

# Load data
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


# Compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
