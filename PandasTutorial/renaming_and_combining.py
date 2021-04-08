import pandas as pd

reviews = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)
pd.set_option('max_rows', 5)

""" RENAMING """

# RENAME() function - change index / column names
print("---------- RENAME FUNCTION FOR COLUMN ----------")
reviews = reviews.rename(columns={'points': 'score'})
print(reviews)

print("---------- RENAME FUNCTION FOR INDEX ----------")
reviews = reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
print(reviews)

# RENAME_AXIS function - change row index and column index
print("---------- RENAME ROW INDEX AND COLUMN INDEX ----------")
reviews = reviews.rename_axis('wines', axis='rows').rename_axis('fields', axis='columns')
print(reviews)


""" COMBINING """

# CONCAT() function - useful when combining
# DATAFRAMES or SERIES with the same COLUMN FIELDS
print("---------- CONCAT TWO DATAFRAMES COMMON COLUMN ----------")
canadian_youtube = pd.read_csv("CAvideos.csv")
british_youtube = pd.read_csv("GBvideos.csv")
conc = pd.concat([canadian_youtube, british_youtube])
print(conc)


# JOIN() function - useful when combining different DATAFRAME objects which have an index in common
print("---------- JOIN TWO DATAFRAMES COMMON INDEX ----------")
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

joined = left.join(right, lsuffix='_CAN', rsuffix='_UK')
print(joined)