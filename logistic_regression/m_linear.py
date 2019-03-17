import numpy as np
from numpy import *
import matplotlib.pyplot as plt
def load_data():
    n = 100
    X = [[1, 0.005*xi] for xi in range(1,n)]
    Y = [2*xi[1] for xi in X]
    return X,Y

def grad_descent(X, Y):
    X = mat(X)
    Y = mat(Y)
    row, col = shape(X)
    alpha = 0.01
    maxIter = 100
    W = ones((1,col))
    for k in range(maxIter):
        W = W + alpha* (Y-W*X.transpose())*X
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
    y2 = W * xM.transpose()

    y22 = [y2[0,i] for i in range(y2.shape[1])]
    plt.plot(x, y22, "o")
    plt.show()
