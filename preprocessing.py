import numpy as np
import pandas as pd
import matplotlib as plt
def clean_num(column_names,data):
    for col in column_names:
        median = data[col].median()
        data[col].fillna(median, inplace=True)