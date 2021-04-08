import pandas as pd
import numpy as np

grades = pd.read_csv('grades.csv')
reviews = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)
# pd.set_option("display.max_rows", 5)

""" GROUPWISE ANALYSIS """

# GROUPBY() function - get the frequency of names [same as VALUE_COUNTS() function]
# returns a SERIES, name as index
print("---------- GROUPBY() function----------")
group1 = grades.groupby("name").name.count()
print(group1)

# GROUPBY() function to get the minimum grade per name
# returns a SERIES, name as index
group2 = grades.groupby("name").grade.min()
print(group2)


# APPLY() function in GROUPBY() function
print("---------- APPLY() and GROUPBY() functions ----------")
# THESE TWO EXAMPLES OUTPUT THE SAME SERIES
# returns a series / column of the first title per winery, winery used as index
print(reviews.groupby('winery').title.first())
# returns a series / column of the first title per winery, winery used as index
print(reviews.groupby('winery').apply(lambda df: df.title.iloc[0]))

# Select the best wine per country province
print("---------- SELECT BEST WINE TITLE GROUPED BY COUNTRY AND PROVINCE ----------")
df = reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])  # type DATAFRAME
best_wine = df.title  # type SERIES
print(best_wine)

# AGG() function and GROUPBY() function lets u perform multiple methods
# Show the LEN, MIN, and MAX PRICES per COUNTRY
print("---------- AGG() and GROUPBY() functions ----------")
print(reviews.groupby(['country']).price.agg([len, min, max]))


""" MULTI-INDEXES """

# GROUPS WITH MULTIPLE PARAMETER
print("---------- MULTI-INDEXES ----------")
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
print(countries_reviewed)

# index attribute
print("---------- GET INDICES OF COUNTRIES REVIEWED ----------")
mi = countries_reviewed.index
print(type(mi))
print(mi)

# RESET_INDEX() function - turns a multi-index into a regular index
print("---------- RESET INDEX ----------")
print(countries_reviewed.reset_index())


""" SORTING """


# SORT_VALUES() function - to sort the dataframe
print("---------- SORT_VALUES() function ----------")
countries_reviewed = countries_reviewed.reset_index()
print(countries_reviewed.sort_values(by='len'))

print("---------- SORT VALUES DESCENDING function ----------")
print(countries_reviewed.sort_values(by='len', ascending=False))

print("---------- SORT INDEX FUNCTION ----------")
print(countries_reviewed.sort_index())

print("---------- SORT MORE THAN 1 COLUMN ----------")
print(countries_reviewed.sort_values(by=['country', 'len']))

print("---------- EXERCISE 1 ----------")
sr = reviews.groupby('taster_twitter_handle').taster_name.count()
print(sr)

print("---------- EXERCISE #2 ----------")
rating = reviews.groupby('price').points.max()
print(rating)

print("---------- EXERCISE #3 ----------")
price_extremes = reviews.groupby('variety').price.agg([min, max])
print(price_extremes)

print("---------- EXERCISE #4 ----------")
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)
print(sorted_varieties)

print("---------- EXERCISE #5 ----------")
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
print(reviewer_mean_ratings)

print("---------- EXERCISE #6 ----------")
country_variety_counts = reviews.groupby(['country', 'variety']).variety.count()
sorted_country_variety_counts = country_variety_counts.sort_values(ascending=False)
print(country_variety_counts)
