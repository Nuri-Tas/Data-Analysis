import pandas as pd
import numpy as np

reviews = pd.read_csv("winemag-data-130k-v2.csv")

missing_data = reviews.isnull().sum()

print(type(missing_data))
print(missing_data.sort_values(ascending=False))
    
percentage_missing = missing_data.index.map(lambda p: missing_data[p] / len(reviews[p]))
print(percentage_missing)

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

```
# This is formatted as code
```

# Data Types and Missing Data

print(reviews[pd.isnull(reviews.country)])

reviews["country"].fillna("unknown")

reviews.fillna("unknown")

reviews.country.replace("Italy", "Italia")

reviews.points.dtype

reviews.points.astype("str")

reviews.loc[reviews.price.isnull()].value_counts

reviews.region_1.fillna("Unknown")
reviews_per_region = reviews.groupby("region_1").agg("max")

reg_1 = reviews.region_1.fillna("Unknown")
print(reg_1.value_counts())

list(reviews)

reviews.rename({"points": "score", "region_1": "Region_1"}, axis="columns")

reviews.rename(index={0: "first_entry", 65498:"last_entry"})

reviews.set_index(["country", "designation"])

reviews.rename_axis("wines", axis="rows").rename_axis("fields", axis="columns")

reviews.country

s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd'])
pd.concat([s1, s2])

s1

df = pd.DataFrame({'month': [1, 4, 7, 10],
                   'year': [2012, 2014, 2013, 2014],
                   'sale': [55, 40, 84, 31]})
df

pd.concat([df, s1])

df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                   'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})

other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                      'B': ['B0', 'B1', 'B2']})

print(df)
print(other)

df.join(other, lsuffix="_A", rsuffix="_B")

countries = reviews.country.unique()

print(countries)

# ***Normalization and Scaling***

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)

original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()

normalized_data = stats.boxcox(original_data)
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()

***Character Encoding***

before = "discipline brings €"

after = before.encode("utf-8", errors="replace")
print(type(after))

after

print(after.decode("utf-8"))

**Inconsistent Data Entries**

reviews.country = reviews.country.fillna("unknown")
unique_countries = reviews.country.unique()
unique_countries.sort()
print(unique_countries)

professors = pd.read_csv("/content/pakistan_intellectual_capital.csv")
professors.Country = professors.Country.str.lower()
professors.Country = professors.Country.str.strip()
professors.Country

import fuzzywuzzy
from fuzzywuzzy import process
import chardet


countries = professors.Country.unique()
matches = fuzzywuzzy.process.extract("southkorea", countries, limit=10,  scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches

def replace_matches(df, column, str_to_match, min_ratio=47):
    countries = df[column].unique()
    matches = fuzzywuzzy.process.extract(str_to_match, countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    rows_with_matches = df[column].isin(close_matches)
    df.loc[rows_with_matches, column] = str_to_match

    print("replace done")

replace_matches(professors, column="Country", str_to_match="south korea")

from google.colab import drive
drive.mount('/content/gdrive')

