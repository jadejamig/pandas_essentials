import pandas as pd

# READ A CSV FILE
print("----------- EXAMPLE 1 -----------")
data1 = pd.read_csv('product.csv')
print(data1)

# Use the shape attribute to check the number of rows and columns of a DATAFRAME
# Result is (row, column)
print(data1.shape)

print("----------- EXAMPLE 2 -----------")
# use the index_col parameter to use a specific column as an index
data2 = pd.read_csv('product.csv', index_col=0)
print(data2)
print(data2.shape)
