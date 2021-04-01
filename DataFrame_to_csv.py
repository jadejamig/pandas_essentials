import pandas as pd

# Write a DataFrame into a CSV file
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Use the to_csv method, the parameter is the filename of the CSV file
animals.to_csv('cows_and_goats.csv')
print('Done!')