import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
data = np.array(data)
data = np.random.shuffle(data)
m, n = data.shape
ytrain = data['label']
xtrain = data.drop(['label'], axis = 1)

# note that there are n features and hence no. of w needed is n
w1 = np.random.randn(10, n-1)
b1 = np.random.randn(10, 1)
w2 = np.random.randn(10, 10)
b2 = np.random.randn(10, 1)


def softmax(z):  
    a = np.exp(z) / sum(np.exp(z))
    return a

# turn probi array to actual label
def onehot(y):
    onehoty = np.zeros((len(y), y.max() + 1))
    onehoty[np.arrange(len(y)), y] = 1
    onehoty = onehoty.T
    return onehoty

def relu(z):
    if z >= 0:
        return z
    else:
        return 0
    
def derivative_relu(z):
    return z > 0
  
def fowardprop(w1,b1,w2,b2,x):
    z1 = np.matmul(x, w1) + b1
    a1 = relu(z1)    
    z2 = np.matmul(a1, w2) + b2
    a2 = softmax(z2)   
    return z1,a1,z2,a2

def backwardprop(z1, a1,  z2, a2, w2,x, y):
    m = y.size
    onehoty = onehot(y)
    dz2 = a2 - onehoty
    dw2 = (1/m) * np.matmul(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, 2)
    dz1 = np.dot(w2.T, dz2) * derivative_relu(z1)
    
    dw1 = (1/m) * np.matmul(dz1, x.T)
    db1 = (1/m) * np.sum(dz1, 2)
    return dw1, db1, dw2, db2

def updateparams(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1= w1 - alpha * dw1
    b1 = b1 - alpha * db1    
    w2 = w2 - alpha * dw2
    b2 = b2 -alpha * db2
    return w1, b1, w1, b2

def gradient_descent(x, y, w1, b1, w2, b2, alpha, num_iters): 
    for i in range(num_iters):
        z1, a1, z2, a2 = fowardprop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backwardprop(z1, a1,  z2, a2, w2, y)
        w1, b1, w1, b2 = updateparams(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        if i%10 == 0:
            print (i)
            print(accuracy(getpredictions(a2), y))
    return w1,b1,w2,b2  

def getpredictions(a2):
    return np.argmax(a2, 0)

def accuracy(predictions, y):
    return np.sum(predictions == y)/ len(y)

w1,b1,w2,b2 = gradient_descent(xtrain, ytrain, w1, b1, w2, b2, 0.01, 100)