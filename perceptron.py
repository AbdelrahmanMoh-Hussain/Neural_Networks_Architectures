import numpy as np
import pandas as pd
from preprocessing import*
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def percep(flist,clist,lr,nm,b):
    df = pd.read_excel('Dry_Bean_Dataset.xlsx')
    filtered_df = df[df['Class'].isin(clist)]
    y=filtered_df['Class']
    x = filtered_df[flist]
    x=normalize(x,flist)
    categories = clist
    label_mapping = {category: label for label, category in enumerate(categories)}
    y = y.map(label_mapping)
    y = y.replace(0, -1)
    if b==1:
        x.insert(0, 'Bias', 1)
        weight_vector = np.random.rand(3) * 0.01
    else:
        weight_vector = np.random.rand(2) * 0.01
    print(weight_vector)
    x_arr = x.to_numpy()
    y_arr = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=40, stratify=y, random_state=42)
    weight_vector=train_perceptron(X_train,y_train,nm,lr,weight_vector)
    er=test_perceptron(X_test,y_test,weight_vector,b,flist)
    print(er)



def train_perceptron(x,y,epoch,learning_rate,weight_vector):
    samples=x.shape[0]
    for e in range(epoch):
        old_weight=weight_vector
        for i in range(samples):
            xi = x[i]
            yi = np.sign(np.dot(weight_vector, xi))
            if yi==0:
                yi=1
            if yi != y[i]:
                loss = y[i] - yi
                weight_vector += learning_rate * loss * xi

        if np.array_equal(weight_vector, old_weight):
            break
    return weight_vector
def test_perceptron(x,y,weight_vector,bias,features_list):
    samples=x.shape[0]
    error=0
    for i in range(samples):
        xi = x[i]
        yi = np.sign(np.dot(weight_vector, xi))
        if yi!=y[i]:
            error+=1
    if bias == 1:
        feature1 = x[:,1]
        feature2 = x[:,2]
    else:
        feature1 = x[:, 0]
        feature2 = x[:, 1]
    plt.scatter(feature1, feature2, c=y, cmap='viridis', label='Data Points')
    plt.xlabel(features_list[0])
    plt.ylabel(features_list[1])
    if bias==1:
        w1 = weight_vector[1]
        w2 = weight_vector[2]
        b = weight_vector[0]
    else:
        w1 = weight_vector[0]
        w2 = weight_vector[1]
        b=0
    x1_decision_boundary = np.linspace(min(feature1), max(feature1), 100)
    x2_decision_boundary = (-w1 * x1_decision_boundary - b) / w2
    plt.plot(x1_decision_boundary, x2_decision_boundary, 'g-', label='Decision Boundary')
    plt.legend()
    plt.show()
    return error


percep(['MajorAxisLength','roundnes'],['BOMBAY','CALI'],0.01,1000,1)