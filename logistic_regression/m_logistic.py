import numpy as np
from numpy import *
import matplotlib.pyplot as plt
def load_data():
    n = 100
    X = [[1, 0.005*xi] for xi in range(1,n)]
    Y = [2*xi[1] for xi in X]
    return X,Y

def sigmoid(z):
    t = exp(z)
    return t/(1+t)
sigmoid_vec = vectorize(sigmoid)
def grad_descent(X, Y):
    X = mat(X)
    Y = mat(Y)
    row, col = shape(X)
    alpha = 0.01
    maxIter = 5000
    W = ones((1,col))
    V = zeros((row, row), float32)
    for k in range(maxIter):
        L = sigmoid_vec(W*X.transpose())
        for i in range(row):
            V[i,i] = L[0,i]*(L[0,i]-1)
        W = W - alpha*(Y-L)*V*X

    return W


if __name__ == "__main__":
    X,Y = load_data()
    print(X)
    print(Y)
    W = grad_descent(X, Y)
    print(W)

    #画图
    x = [xi[1] for xi in X]
    y = Y
    plt.plot(x, y, "*")

    xM = mat(X)
    y2 = sigmoid_vec(W * xM.transpose())

    y22 = [y2[0,i] for i in range(y2.shape[1])]
    plt.plot(x, y22, "o")
    plt.show()
