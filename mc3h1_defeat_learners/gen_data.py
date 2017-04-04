"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    X = np.random.standard_normal(size=(1000, 2))
    Z = X.sum(axis=1)
    size = X.shape[0]
    Y = Z + np.random.normal(size=size)
    # 1X = np.mgrid[-5:5:0.5,-5:5:0.5].reshape(2,-1).T
    # Y = X[:,0]*X[:,1] + np.random.normal(size = X.shape[0])
    return X, Y


def best4RT(seed=1489683273):
    np.random.seed(seed)
    mu, sigma = 10, 140
    # X1 = np.random.normal(mu, sigma, size=(500,20))
    # mu, sigma = 0, 120
    # X2 = np.random.normal(mu, sigma, size = (500, 20))
    # X = np.concatenate((X1,X2), axis=0)
    # X = np.random.rand(1000,20)
    #X1 = np.random.uniform(0, 10, 1000)
    X1 = np.random.normal(25, 25, 5000)
    X2 = np.zeros(X1.shape[0])

    for i in range(len(X1)):
        if X1[i] > 25:
            X2[i] = np.random.normal(-100, 5, 1)
        else:
            X2[i] = np.random.normal(100, 5,  1)

    X = np.column_stack((X1, X2))
    Y = np.zeros(X1.shape[0])

    for i in range(len(X1)):
        if X1[i] > 25:
            if X2[i] > -100:
                Y[i] = np.random.normal(10, 1, 1)
            else:
                Y[i] = np.random.normal(5, 1, 1)
        else:
            if X2[i]> -100:
                Y[i] = np.random.normal(5, 1, 1)
            else:
                Y[i] = np.random.normal(10, 1, 1)

    # Y = np.random.rand(X.shape[0])


    # Y= np.random.standard_normal(X1.shape[0] + X2.shape[0])
    # Y = np.random.normal(mu, sigma, X1.shape[0] + X2.shape[0])
    # X = np.random.normal(size = (50, 2))
    # Y = 0.8 * X[:,0] + 5.0 * X[:,1]
    return X, Y


if __name__ == "__main__":
    X1, Y1 = best4LinReg(seed=5)
    X2, Y2 = best4RT(seed=5)
    plt.plot(X1, Y1)
    plt.title('Best4LinReg')
    # plt.show()

    plt.plot(X2, Y2)
    plt.title('Best4RandomTree')
    plt.show()
    # print "they call me Tim."
