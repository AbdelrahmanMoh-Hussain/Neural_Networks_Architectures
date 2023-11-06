import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib as plt


def clean_num(x, column_names):
    for col in column_names:
        median = x[col].median()
        x[col].fillna(median, inplace=True)
    return x



def normalize(x, column_names):
    for col in column_names:
        min_value = x[col].min()
        max_value = x[col].max()
        x.loc[:, col] = (x[col] - min_value) / (max_value - min_value)  # edited to avoid warning
    return x
