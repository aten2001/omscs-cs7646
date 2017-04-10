import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spo

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt


def show_histogram():
    np.random.seed(2000)
    y = np.random.standard_normal((1000, 2)).cumsum(axis=0)
    plt.figure(figsize=(7, 5))
    plt.hist(y, label=['1st', '2nd'], bins=25)
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('histogram')


def show_scatter_with_color_bar():
    np.random.seed(2000)
    y = np.random.standard_normal((1000, 2)).cumsum(axis=0)
    c = np.random.randint(0, 10, len(y))
    plt.figure(figsize=(7, 5))
    plt.scatter(y[:, 0], y[:, -1], c=c, marker='o')
    plt.colorbar()
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')


def show_scatter_with_scatter_func():
    np.random.seed(2000)
    y = np.random.standard_normal((1000, 2)).cumsum(axis=0)
    plt.figure(figsize=(7, 5))
    plt.scatter(y[:, 0], y[:, -1], marker='o')
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')


def show_scatter_basic():
    np.random.seed(2000)
    y = np.random.standard_normal((1000, 2)).cumsum(axis=0)
    plt.figure(figsize=(7, 5))
    plt.plot(y[:, 0], y[:, 1], 'ro')
    plt.grid(True)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.title('Scatter Plot')


def show_with_different_plot_types():
    np.random.seed(2000)
    y = np.random.standard_normal((20, 2)).cumsum(axis=0)
    plt.figure(figsize=(9, 4))
    # 1 row, 2 columns, 1st plt
    plt.subplot(121)
    plt.plot(y[:, 0], lw=1.5, label='1st')
    plt.plot(y[:, 0], 'ro')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.title('1st Data Set')
    plt.subplot(122)
    plt.bar(np.arange(len(y)), y[:, 1], width=0.5, color='g', label='2nd')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('index')
    plt.title('2nd data set')


def show_using_sub_plots():
    np.random.seed(2000)
    y = np.random.standard_normal((20, 2)).cumsum(axis=0)
    plt.figure(figsize=(7, 5))
    plt.subplot(211)
    plt.plot(y[:, 0], lw=1.5, label='1st')
    plt.plot(y[:, 0], 'ro')
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
    plt.plot(y[:, 0], 'b', lw=1.5, label='1st')
    plt.plot(y[:, 0], 'ro')
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


def SMA(df, n):
    # pd.Series.rolling(window=10, center=False).mean()
    SMA = pd.Series(pd.rolling_mean(df['Adj Close'], n), name='SMA')
    df = df.join(SMA)
    return df


# Exponential Moving Average
def EMA(df, n):
    EMA = pd.Series(pd.ewma(df['Adj Close'], span=n, min_periods=n - 1), name='EMA')
    df = df.join(EMA)
    return df


# Bollinger Bands
def BBANDS(df, n):
    MA = pd.Series(pd.rolling_mean(df['Adj Close'], n))
    MSD = pd.Series(pd.rolling_std(df['Adj Close'], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)
    return df


def show_prices_sma(prices, window):
    price_with_sma = pd.rolling_mean(prices, window)


def show_sma():
    unique_symbols = ['AAPL']
    dates = pd.date_range('2010-01-01', '2012-12-31')
    prices_df = get_data(symbols=unique_symbols,
                         dates=dates,
                         addSPY=True,
                         colname='Adj Close')

    prices_df = prices_df.rename(columns={'AAPL': 'Adj Close'})
    SMA_AAPL = SMA(prices_df, 20)
    SMA_AAPL = SMA_AAPL.dropna()
    SMA = SMA_AAPL['SMA']

    plt.figure(figsize=(9, 5))
    plt.plot(prices_df['Adj Close'], lw=1, label='AAPL Prices')
    plt.plot(SMA, 'g', lw=1, label='20 day SMA (green)')
    plt.legend(loc=2, prop={'size': 11})
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()


def get_rolling_mean(values, window):
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    return pd.rolling_std(values, window=window)


def get_bollinger_bands(rm, rstd):
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def show_bollinger_band(prices_df, symbol):
    rm = get_rolling_mean(prices_df[symbol], window=20)
    rstd = get_rolling_std(prices_df[symbol], window=20)
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    # Plt raw values, rolling mean and Bollinger Bands
    ax = prices_df[symbol].plot(title="Bollinger Bands", label=symbol)
    rm.plot(label='Rolling Mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    lower_indicator = lower_band - prices_df[symbol]
    upper_indicator = prices_df[symbol] - upper_band

    lower_indicator.plot(label='lower bound  - price', ax=ax)
    upper_indicator.plot(label='price - upper bound', ax=ax)

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    # show_difference_scaling()
    # show_using_two_axes()
    # show_using_sub_plots()
    # show_with_different_plot_types()
    # show_scatter_basic()
    # show_scatter_with_scatter_func()
    # show_scatter_with_color_bar()
    # show_histogram()

    unique_symbols = ['AAPL']
    start_dt = '2010-01-04'
    last_dt = '2011-04-26'
    prices_df = get_data(symbols=unique_symbols,
                         dates=pd.date_range(start_dt, last_dt),
                         addSPY=True,
                         colname='Adj Close')

    # prices_df = prices_df.rename(columns={'AAPL': 'Adj Close'})
    # SMA_AAPL = SMA(prices_df, 20)
    # SMA_AAPL = SMA_AAPL.dropna()
    # SMA = SMA_AAPL['SMA']

    # plt.figure(figsize=(9, 5))
    # plt.plot(prices_df['Adj Close'], lw=1, label='AAPL Prices')
    # plt.plot(SMA, 'g', lw=1, label='20 day SMA (green)')
    # plt.legend(loc=2, prop={'size': 11})
    # plt.grid(True)
    # plt.setp(plt.gca().get_xticklabels(), rotation=30)
    # plt.show()

    show_bollinger_band(prices_df, 'AAPL')
    plt.show()
