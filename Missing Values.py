import pandas as pd
import numpy as np

reviews = pd.read_csv("winemag-data-130k-v2.csv")


missing_data = reviews.isnull().sum()

print(type(missing_data))
print(missing_data.sort_values(ascending=False))


for element in missing_data.index:
    print(element)
    
# calculate the percentage of missind data in each seperate row

percentage_missing = missing_data.index.map(lambda p: missing_data[p] / len(reviews[p]))
print(percentage_missing)

# the total percentage of missing data

total_amount = np.product(reviews.shape)
total_miss = missing_data.sum()
percentage = total_miss/total_amount*100
print(percentage)

dropped = reviews.dropna(axis=1)


# the columns that dropped

print(dropped)

#keep in mind that axis=0 as we want nan's to be filled with the value of the next row
print(reviews.fillna(method="bfill", axis=0))
print(reviews.size)
print(len(reviews))


