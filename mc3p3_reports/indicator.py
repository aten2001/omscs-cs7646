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


def z_score_df(df):
    mean = df.mean()
    std = df.std()
    z_scored = (df - mean) / std
    return z_scored


def MACD(df, symbol, n_fast, n_slow):
    EMAfast = pd.Series(pd.ewma(df[symbol], span=n_fast, min_periods=n_slow - 1))
    EMAslow = pd.Series(pd.ewma(df[symbol], span=n_slow, min_periods=n_slow - 1))
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(pd.ewma(MACD, span=9, min_periods=8), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def compute_normalized(df):
    normalized = df[0:] / df.ix[0]
    return normalized


# Momentum
def MOM(df, symbol, n):
    M = pd.Series(df[symbol].diff(n), name='Momentum_' + str(n))
    df = df.join(M)
    return df


# Relative Strength Index
def RSI(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(pd.ewma(UpI, span=n, min_periods=n - 1))
    NegDI = pd.Series(pd.ewma(DoI, span=n, min_periods=n - 1))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df = df.join(RSI)
    return df


def show_rsi(prices_df, symbol):
    print 'a'


def show_momentum(prices_df, symbol):
    momentum = MOM(prices_df, symbol, 20)
    ema = pd.Series(pd.ewma(prices_df[symbol], span=9, min_periods=8), name='9 day EMA')
    ema_zscored = z_score_df(ema)
    print ema


    rstd = get_rolling_std(prices_df[symbol], window=20)
    plt.subplot(211)
    prices_normalized = compute_normalized(prices_df[symbol])
    ax1 = prices_normalized.plot(title="Momentum", color='r', label=symbol)

    plt.xlabel('Date')
    plt.ylabel('Price')
    ax1.legend(loc='upper left')
    plt.grid(True)

    momentum_zscored = z_score_df(momentum)
    ax2 = ax1.twinx()
    momentum_zscored['Momentum_20'].plot(label='Momentum', color='b', ax=ax2)

    ax2.set_ylabel("Momentum / Vol")
    rstd_zscored = z_score_df(rstd)
    rstd_zscored.plot(color='g', label='Vol', ax=ax2)
    ax2.legend(loc='upper right')

    # second plot
    #print momentum_zscored
    diff = momentum['Momentum_20'] / rstd
    #print diff
    diff_zscored = z_score_df(diff)
    plt.subplot(212)
    ax3 = diff_zscored.plot(title='Momentum vs Vol and EMA(9)', label='Momentum vs Vol', color='c')
    ema_zscored.plot(label='9 day EMA', color='y', ax=ax3)
    plt.grid(True)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Momentum vs Vol")
    plt.show()


def show_macd(prices_df, symbol):
    macd = MACD(prices_df, symbol, 12, 26)
    plt.subplot(211)
    # macd.plot(label='MACD(12,26,9)')
    prices_normalized = compute_normalized(prices_df[symbol])
    ax1 = prices_normalized.plot(title="MACD(12,26,9)", color='r', label=symbol)
    plt.xlabel('Date')
    plt.ylabel('Price')
    ax1.legend(loc='upper left')
    plt.grid(True)

    # second plot
    macd_zscored = z_score_df(macd['MACD_12_26'])
    macd_signed = z_score_df(macd['MACDsign_12_26'])
    macd_diff = z_score_df(macd['MACDdiff_12_26'])

    ax2 = ax1.twinx()
    macd_zscored.plot(label='MACD_12_26', color='b', ax=ax2)
    macd_signed.plot(label='9 day EMA', color='g', ax=ax2)
    ax2.set_ylabel("MACD")
    ax2.legend(loc='upper right')

    plt.subplot(212)
    macd_diff.plot(title='MACD vs Signal', label='MACD vs Signal', color='c')
    plt.grid(True)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("MACD")
    plt.show()


def show_bollinger_band(prices_df, symbol):
    rm = get_rolling_mean(prices_df[symbol], window=20)
    rstd = get_rolling_std(prices_df[symbol], window=20)
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    # Plt raw values, rolling mean and Bollinger Bands
    plt.subplot(211)
    ax = prices_df[symbol].plot(title="Bollinger Bands", label=symbol)
    rm.plot(label='Rolling Mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)
    plt.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')

    lower_indicator = lower_band - prices_df[symbol]
    upper_indicator = prices_df[symbol] - upper_band

    # sell when upper indicator is positive
    # buy when lower indicator is positive
    lower_indicator_zscored = z_score_df(lower_indicator)
    upper_indicator_zscored = z_score_df(upper_indicator)

    # second plot
    plt.subplot(212)
    plt.legend(loc=0)

    ax2 = lower_indicator_zscored.plot(label='lower bound  - price')
    upper_indicator_zscored.plot(label='price - upper bound', ax=ax2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Indicator")
    ax2.legend(loc='upper left')
    plt.grid(True)
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
    start_dt = '2008-01-01'
    last_dt = '2008-04-15'
    prices_df = get_data(symbols=unique_symbols,
                         dates=pd.date_range(start_dt, last_dt),
                         addSPY=True,
                         colname='Adj Close')

    show_bollinger_band(prices_df, 'AAPL')

    show_macd(prices_df, 'AAPL')
    show_momentum(prices_df, 'AAPL')
