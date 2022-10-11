import pandas as pd
import numpy as np
import os
#load data to inspect
path = os.path.abspath(os.getcwd())
df = pd.read_csv(path + "/../data/census.csv")

# clear spaces from column names of the df
df.columns = [ column_name.strip() for column_name in df.columns]
#remove spaces from each row of a categorical column
cat_features = df.select_dtypes(include=['object']).columns
df[cat_features] = df[cat_features].apply(lambda x: x.str.strip())

df.to_csv(path + "/../data/clean_census.csv", index=False)