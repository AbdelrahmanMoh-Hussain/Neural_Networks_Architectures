import numpy as np
import pandas as pd
def percep(flist,clist,lr,nm,b):
    df = pd.read_excel('Dry_Bean_Dataset.xlsx')
    filtered_df = df[df['Class'].isin(clist)]
    y=filtered_df['Class']
    x = filtered_df[flist]
    if b==1:
        x.insert(0, 'Bias', 1)
        weight_vector = np.random.rand(3) * 0.01
    else:
        weight_vector = np.random.rand(2) * 0.01
    #print(weight_vector)
    #print(y)
    #print(x)

percep(['Area','Perimeter'],['BOMBAY','CALI'],2,2,1)