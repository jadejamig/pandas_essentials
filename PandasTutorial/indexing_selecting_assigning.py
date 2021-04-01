import pandas as pd

products = pd.read_csv('product.csv')
# Set the max rows to 5 only when printing
pd.set_option('max_rows',5)
print("---------- Products ----------")
print(products)

# Accessing a column / series in a DATAFRAME
# You can use either the DOT NOTATION or the BRACKET NOTATION
print("---------- Accessing Column using dot and bracket notation----------")
col1_dotNotation = products.ProductA  # DOT NOTATION
col1_brackets = products["ProductA"]  # BRACKET NOTATION
print(col1_dotNotation)
print(col1_brackets)

# Accessing an element in a column / series
print("---------- Accessing elements ----------")
element1 = col1_dotNotation[0]
element2 = col1_brackets[0]
print(element1)
print(element2)

# INDEX-BASED SELECTION (iloc) - selecting data based on its numerical position
print("---------- Accessing Row Using iloc ----------")
# Select the first row of data in the DATAFRAME
row1_iloc = products.iloc[0]
print(row1_iloc)

# Select the last 3 rows in the DATAFRAME
row2_iloc = products.iloc[-3:]
print(row2_iloc)

# Selecting a column using iloc[ROW, COLUMN]
print("---------- Accessing Column using iloc ----------")
col1_iloc = products.iloc[:, 0]
print(col1_iloc)

# Select the first column of the first three rows
col2_iloc = products.iloc[:3, 0]
print(col2_iloc)

# Select columns of Specific rows using a list in iloc
# Select first column of rows 0, 3, and 6
col3_iloc = products.iloc[[0, 3, 6], 0]
print(col3_iloc)

# LABEL-BASED SELECTION (loc)
