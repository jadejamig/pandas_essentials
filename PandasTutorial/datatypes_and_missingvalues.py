import pandas as pd

# Investigate data types in a DataFrame or Series
# Learn how to find and replace entries

reviews = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)

""" Dtypes """

# Dtype/s - data type for a column or series
print("---------- dtypes attribute of a DataFrame ----------")
print(reviews.dtypes)

print("---------- dtype attribute of a column / series ----------")
print(reviews.price.dtype)
print(reviews.index.dtype)

# ASTYPE() function - convert a column from one type to another
col = reviews.points.astype('float64')  # convert from int64 to float 64
print(col)

""" MISSING DATA """

# PD.ISNULL() / PD.NOTNULL() function - select NaN values
# Missing data has a value of NaN which means "Not a Number"
# NaN values are float64 dtype
print("---------- SELECTING MISSING / NaN VALUES ----------")
print(reviews[pd.isnull(reviews.country)])

# FILLNA() function - replace missing values
print("---------- REPLACE MISSING / NaN VALUES ----------")
print(reviews.region_2.fillna('unknown'))

# REPLACE() function - replace not-null values
print("---------- REPLACE NOT-NULL VALUES ----------")
print(reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino"))

# EXERCISES
print("---------- EXERCISE 1 ----------")
print(reviews.points.dtype)

print("---------- EXERCISE 2 ----------")
print(reviews.points.astype('str'))

print("---------- EXERCISE 3 ----------")
n = pd.isnull(reviews.price)
n_missing_prices = n.sum()
print(n_missing_prices)

print("---------- EXERCISE 4 ----------")
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)
print(reviews_per_region)

