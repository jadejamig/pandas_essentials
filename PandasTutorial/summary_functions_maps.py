import pandas as pd
import numpy as np

""" SUMMARY FUNCTIONS - restructure the data in some useful way """

# DESCRIBE() method for numerical values returns a high-level summary
print("---------- Describe method for numerical values ----------")
products = pd.read_csv('product.csv')
print(products.describe())

grades = pd.read_csv('grades.csv')
print("---------- Describe method for text values ----------")
print(grades.name.describe())
print(grades.subject.describe())

# MEAN() function
print("---------- Getting the mean of a column ----------")
print(grades.grade.mean())

# UNIQUE() function - used to see a list of unique values
print("---------- UNIQUE() function Get unique values ----------")
print(grades.name.unique())

# VALUE_COUNTS() function - to see a list of frequency of unique values
print("---------- VALUE_COUNTS() function Get unique values frequency ----------")
print(grades.name.value_counts())

""" MAPS - takes one set of values and "maps" them to another set of values """

# MAP() function - will only work on a series, returns a series
print("---------- MAP() function ----------")
mean = grades.grade.mean()
grade_mapped = grades.grade.map(lambda p: p - mean if p < mean else p)
print(grade_mapped)


# APPLY() function - used to transform a whole dataframe
# by calling a custom method on each row


def remean_grade(row):
    grade_mean = grades.grade.mean()
    row.grade = row.grade - grade_mean
    return row


print("---------- APPLY() function ----------")
grade_applied = grades.apply(remean_grade, axis='columns')
print(grade_applied)

print("---------- COMBINING COLUMNS ----------")
print(grades.name + "-" + grades.subject)


# APPLY() function that return a series
print("---------- APPLY() function returning a SERIES / COLUMN ----------")
def stars(row):
    if row.grade >= 95:
        return "3 stars"
    elif row.grade >= 90:
        return "2 stars"
    else:
        return "1 star"


star_ratings = grades.apply(stars, axis='columns')
print(star_ratings)
