import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib as plt
def clean_num(column_names,data):
    for col in column_names:
        median = data[col].median()
        data[col].fillna(median, inplace=True)
def normalize(x,column_names):
    for col in column_names:
        min_value = x[col].min()
        max_value = x[col].max()
        x[col] = (x[col] - min_value) / (max_value - min_value)
    return x


