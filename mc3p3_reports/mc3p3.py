import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spo

def show_scatter_with_color_bar():
    np.random.seed(2000)
    y = np.random.standard_normal((1000, 2)).cumsum(axis=0)
    c = np.random.randint(0,10,len(y))
    plt.figure(figsize=(7, 5))
    plt.scatter(y[:, 0], y[:, -1], c=c,marker='o')
    plt.colorbar()
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')

def show_scatter_with_scatter_func():
    np.random.seed(2000)
    y = np.random.standard_normal((1000, 2)).cumsum(axis=0)
    plt.figure(figsize=(7, 5))
    plt.scatter(y[:,0], y[:,-1], marker='o')
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')

def show_scatter_basic():
    np.random.seed(2000)
    y = np.random.standard_normal((1000, 2)).cumsum(axis=0)
    plt.figure(figsize=(7,5))
    plt.plot(y[:,0], y[:,1], 'ro')
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')

def show_with_different_plot_types():
    np.random.seed(2000)
    y = np.random.standard_normal((20, 2)).cumsum(axis=0)
    plt.figure(figsize=(9,4))
    #1 row, 2 columns, 1st plt
    plt.subplot(121)
    plt.plot(y[:,0], lw=1.5, label='1st')
    plt.plot(y[:,0], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.title('1st Data Set')
    plt.subplot(122)
    plt.bar(np.arange(len(y)), y[:,1], width=0.5, color='g', label='2nd')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.title('2nd data set')


def show_using_sub_plots():
    np.random.seed(2000)
    y = np.random.standard_normal((20, 2)).cumsum(axis=0)
    plt.figure(figsize=(7,5))
    plt.subplot(211)
    plt.plot(y[:,0], lw=1.5, label='1st')
    plt.plot(y[:,0], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.ylabel('value')
    plt.title('A Simple Plot')
    plt.subplot(212)
    plt.plot(y[:, 1], 'g', lw=1.5, label='2nd')
    plt.plot(y[:, 1], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value')


def show_using_two_axes():
    np.random.seed(2000)
    y = np.random.standard_normal((20, 2)).cumsum(axis=0)
    fig, ax1 = plt.subplots()
    plt.plot(y[:,0], 'b', lw=1.5, label='1st')
    plt.plot(y[:,0], 'ro')
    plt.grid(True)
    plt.legend(loc=8)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value 1st')
    plt.title('A Simple Plot')
    ax2 = ax1.twinx()
    plt.plot(y[:, 1], 'g', lw=1.5, label='2nd')
    plt.plot(y[:, 1], 'ro')
    plt.legend(loc=0)
    plt.ylabel('value 2nd')


def show_difference_scaling():
    np.random.seed(2000)
    y = np.random.standard_normal((20, 2)).cumsum(axis=0)
    y[:, 0] = y[:, 0] * 100
    plt.figure(figsize=(7, 4))
    plt.plot(y[:, 0], lw=1.5, label='1st')
    plt.plot(y[:, 1], lw=1.5, label='2nd')
    plt.plot(y, 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.title('A Simple Plot')


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    #show_difference_scaling()
    #show_using_two_axes()
    #show_using_sub_plots()
    #show_with_different_plot_types()
    #show_scatter_basic()
    #show_scatter_with_scatter_func()
    show_scatter_with_color_bar()
    plt.show()
    # plt.show()
