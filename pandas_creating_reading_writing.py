import pandas as pd

""" CREATING DATA 
    There are two core objects in pandas: the DataFrame and the Series.
"""

# DATAFRAME - is a Table
print("---------- DATAFRAME SECTION ----------")

# EXAMPLE DataFrame1
dataFrame1 = pd.DataFrame({"Yes": [50, 21], "No": [131, 2]})
print(dataFrame1)

# EXAMPLE DataFrame2
dataFrame2 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
print(dataFrame2)

# EXAMPLE DataFrame3
# Specify the row label in a DataFrame
# pass a dictionary and a list which represents the row label (Product A and Product B)
dataFrame3 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
print(dataFrame3)

# SERIES - is a sequence of data values, a Series is a list.
# A series is a single column of a DATAFRAME
print("---------- SERIES SECTION ----------")

# EXAMPLE series1
series1 = pd.Series([1, 2, 3, 4, 5])
print(series1)

# EXAMPLE series2 - assign INDEX and series NAME
series2 = pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')