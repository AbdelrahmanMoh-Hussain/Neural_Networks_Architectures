import numpy as np
import pandas as pd
import math
from preprocessing import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
def propagation(number_of_hidden_layers, number_of_neurons,learing_rate, epochs, bias,function):
    df = pd.read_excel('Dry_Bean_Dataset.xlsx')
    y = df['Class']
    x = df.drop(columns=['Class'])
    flist = x.columns.tolist()
    x = clean_num(x, flist)
    x = normalize(x, flist)
    mapping = {'BOMBAY': [1, 0, 0], 'CALI': [0, 1, 0], 'SIRA': [0, 0, 1]}
    y = y.map(mapping)
    # print(y)
    num_layers = number_of_hidden_layers+2
    neurons_count=[5]
    for i in number_of_neurons:
        neurons_count.append(i)
    neurons_count.append(3)
    # print(neurons_count)
    weights = [np.random.randn(neurons_count[i+1], neurons_count[i])*0.01 for i in range(num_layers - 1)]
    biases = [np.random.randn(1, neurons_count[i + 1])*0.01 for i in range(num_layers - 1)]
    # print(weights[0][0]) first dimension is for the layer weights[0] will get the weights for all neurons in this layer
    # weights[0][0] gets the weights for the first neuron in first layer
    # print(biases[0][0][1]) first dimension for layer, second is constant with 0 always, third for neuron
    x_arr = x.to_numpy()
    y_arr = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=40, stratify=y, random_state=42)
    train(neurons_count, num_layers, weights, biases, X_train, y_train, epochs)

def train(neurons_count,layers_count,weights,biases,x,y,epoch):
    samples = x.shape[0]
    for e in range(epoch):
        for i in range(samples):
            z=forward(weights,biases,x[i])
            #print(z)
            backward(weights,biases,layers_count,neurons_count,epoch,0,z,x[i],y[i], 5)
            break
        break


def forward(weights, biases, sample):
    z=[]
    for i in range(len(weights)):
        z.append([])
        for j in range(len(weights[i])):
            if  i==0:
                net=np.dot(weights[i][j],sample)+biases[i][0][j]
            else:
                net = np.dot(weights[i][j], z[i-1]) + biases[i][0][j]
            sig=1 / (1 + np.exp(-net))
            z[i].append(sig)
    return z

def backward(weights, biases, layers_count, neurons_count,epochs ,activation_function,z,x,y, learning_rate):
    error = [[] for i in range(layers_count)]
    for layer in reversed(range(0,layers_count)):
        for neuron in range(0,neurons_count[layer]):
            if(layer == layers_count - 1): #1st case: hidden to Output [[ (d-y)*f`(net) ]]
                error[layer].append((y[neuron] - z[layer - 1][neuron]) * derivative(z[layer - 1][neuron], activation_function))

            elif layer == 0: #update weights
                update_weights(weights, error, learning_rate, neurons_count, z, x)
                break

            else: #1st case: [[ f`(net) * sum(error[k]*weight[k][j] ]]
                derivative_func = derivative(z[layer - 1][neuron], activation_function)
                summation = 0
                for k in range(0,neurons_count[layer + 1]):
                    summation += error[layer + 1][k] * weights[layer][k][neuron]
                error[layer].append(derivative_func * summation)

    #forward(weights, biases, x)

def update_weights(weights_list, error_list, learning_rate, neurons_count,z,x):
    print("Before")
    print(weights_list)
    print(z)
    for layer_index in range(0, 3):
        print("Layer # " + str(layer_index))
        for i in range(neurons_count[layer_index]):
            print("i = " + str(i))
            for k in range(0, neurons_count[layer_index + 1]):
                print("K = " + str(k))
                if layer_index == 0:
                    print("HERE X = " + str(x[i]))
                    weights_list[layer_index][k][i] += learning_rate * error_list[layer_index + 1][k] * x[i]
                else:
                    weights_list[layer_index][k][i] += learning_rate * error_list[layer_index + 1][k] * z[layer_index - 1][i]
        print(weights_list)
def derivative(value, activation_function):
    if activation_function == 0:
        return value * (1 - value)
    else:
        return 1 - pow(math.tanh(value),2)

propagation(2,[3,4],0.1,1000,1,1)