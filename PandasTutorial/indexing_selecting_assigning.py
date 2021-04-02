import pandas as pd

products = pd.read_csv('PandasTutorial/product.csv')
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

# Select multiple columns of multiple rows using a list in iloc
# select the first 2 columns of rows 0, 3, and 6
col4_iloc = products.iloc[[0, 3, 6], [0, 1]]  # You can also use ":2" instead of "[0, 1]"
print(col4_iloc)

# LABEL-BASED SELECTION (loc) - selecting data based on index value
print("---------- Accessing Elements using loc ----------")
element1_loc = products.loc[0, 'ProductA']
print("Element 0 in ProductA column: ", element1_loc)

# Selecting multiple specific columns using loc (column names)
print("---------- Accessing Specific Columns using loc ----------")
col1_loc = products.loc[:2, ['ProductA', 'ProductC']]
print(col1_loc)

# Selecting ROWS if the INDEX is a STRING
# Selecting the productA and productC columns of the row 1 to 5
print("---------- Accessing Specific Rows using loc ----------")
col2_loc = products.loc['1': '5', ['ProductA', 'ProductC']]
print(col2_loc)


# MANIPULATING THE INDEX
# The set_index() method can be used to transform
# a DATAFRAME column into the DATAFRAME index
print("---------- MANIPULATING THE INDEX ----------")
indexed_df = products.set_index("ProductC")
print(indexed_df)

# BONUS!! Selecting the mean of the ProductA and ProductC columns
# using describe method and loc for index name selection
print("---------- SELECTING THE MEAN OF SPECIFIC COLUMNS (loc) ----------")
stats = products.describe()
mean = stats.loc['mean', ['ProductA', 'ProductC']]
print(mean)

# CONDITIONAL SELECTION - returns a series / list
# with TRUE or FALSE values depending on the condition
print("---------- CONDITIONAL SELECTION ----------")
conditional_selection = products["ProductA"] == 30
print(conditional_selection)

# CONDITIONAL SELECTION in loc - returns a ROW/S
print("---------- CONDITIONAL SELECTION IN loc----------")
conditional_selection_loc1 = products.loc[products["ProductA"] == 30]
print(conditional_selection_loc1)

# CHECK IF ELEMENT OF A LIST
# USING isin()
print("---------- CONDITIONAL SELECTION IN loc Using isin----------")
conditional_selection_loc2 = products.loc[products["ProductA"].isin([35, 45, 10])]
print(conditional_selection_loc2)

# CONDITIONAL SELECTION in loc with specific column
print("---------- CONDITIONAL SELECTION IN loc With Specific Column----------")
conditional_selection_loc3 = products.loc[products["ProductA"] == 30, ['ProductA', 'ProductC']]
print(conditional_selection_loc3)

# COMPOUND CONDITIONAL SELECTION in loc
print("---------- COMPOUND CONDITIONAL SELECTION IN loc----------")
comp_conditional_selection = products.loc[(products["ProductA"] < 30) & (products["ProductB"] > 30)]
print(comp_conditional_selection)

# CONDITIONAL SELECTION checking if NULL or NOT NULL
print("---------- CONDITIONAL SELECTION NULL or NOT NULL ----------")
conditional_not_null = products.loc[products["ProductA"].notnull()]
print(conditional_not_null)

"""ASSIGNING DATA"""
print("---------- ASSIGNING DATA TO COLUMNS ----------")
products.iloc[:, :2] = "wow"
print(products)

# You can also assign an iterable value
print("---------- ASSIGNING ITERABLE VALUE ----------")
products['index_backwards'] = range(len(products), 0, -1)
print(products)

