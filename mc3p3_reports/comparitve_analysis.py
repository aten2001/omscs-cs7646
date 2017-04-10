"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
import datetime
import math
import time
import RTLearner as rtLearner
import rule_based as rule_based
import ml_based as ml_based


def author():
    return 'jlee3259'  # replace tb34 with your Georgia Tech username.


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
def calc_ema(df, symbol, n):
    EMA = pd.Series(pd.ewma(df[symbol], span=n, min_periods=n - 1), name='EMA')
    df = df.join(EMA)
    return df


# Bollinger Bands
def calc_bb(df, symbol, n):
    MA = pd.Series(pd.rolling_mean(df[symbol], n))
    MSD = pd.Series(pd.rolling_std(df[symbol], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='Bollinger_Lower')
    df = df.join(B1)
    b2 = (df[symbol] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger_Upper')
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


def calc_macd(df, symbol, n_fast, n_slow):
    EMAfast = pd.Series(pd.ewma(df[symbol], span=n_fast, min_periods=n_slow - 1))
    EMAslow = pd.Series(pd.ewma(df[symbol], span=n_slow, min_periods=n_slow - 1))
    MACD = pd.Series(EMAfast - EMAslow, name='MACD')
    MACDsign = pd.Series(pd.ewma(MACD, span=9, min_periods=8), name='MACDsign')
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff')
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def compute_normalized(df):
    normalized = df[0:] / df.ix[0]
    return normalized


def calc_std(df, symbol, n):
    df = df.join(pd.Series(pd.rolling_std(df[symbol], n), name='STD'))
    return df


# Momentum
def calc_momentum(df, symbol, n):
    M = pd.Series(df[symbol].diff(n), name='Momentum')
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
    momentum = calc_momentum(prices_df, symbol, 20)
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
    # print momentum_zscored
    diff = momentum['Momentum_20'] / rstd
    # print diff
    diff_zscored = z_score_df(diff)
    plt.subplot(212)
    ax3 = diff_zscored.plot(title='Momentum vs Vol and EMA(9)', label='Momentum vs Vol', color='c')
    ema_zscored.plot(label='9 day EMA', color='y', ax=ax3)
    plt.grid(True)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Momentum vs Vol")
    plt.show()


def show_macd(prices_df, symbol):
    macd = calc_macd(prices_df, symbol, 12, 26)
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


def compute_daily_returns(daily_pos_df):
    """Compute and return the daily return values."""
    # compute daily returns for row 1 onwards
    # df_temp = daily_pos_df.sum(axis=1)
    daily_returns = (daily_pos_df / daily_pos_df.shift(1)) - 1
    # daily_returns.ix[0, :] = 0  # set daily returns for row 0 to 0
    return daily_returns


def compute_normalized(df):
    normalized = df[0:] / df.ix[0]
    return normalized


def compute_alloced(normalized, syms, allocs):
    alloced = normalized[syms] * allocs
    return alloced


def compute_pos_vals(alloced, sv):
    return alloced * sv


def compute_avg_daily_return(daily_returns):
    return daily_returns.mean()


def compute_sddr(daily_returns):
    return daily_returns.std()


def compute_cr(port_val):
    return (port_val[-1] / port_val[0]) - 1


def compute_sr(daily_returns, rf, sf):
    return ((daily_returns - rf).mean() / daily_returns.std(ddof=1)) * np.sqrt(sf)


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    # this is the function the autograder will call to test your code
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True,
                            usecols=['Date', 'Symbol', 'Order', 'Shares'])

    start_dt = orders_df.index[0]
    last_dt = orders_df.index[-1]
    unique_symbols = orders_df.drop_duplicates('Symbol')['Symbol'].tolist()

    # Step 1 - Prices Table
    prices_df = get_data(unique_symbols, pd.date_range(start_dt, last_dt), True, 'Adj Close')
    prices_df['Cash'] = 1

    # Step 2 - Trades Table
    trades_df = prices_df.copy()
    for symbol in unique_symbols:
        trades_df[symbol + '_Trade'] = 0
    trades_df['Cash_Trade'] = 0

    for index, row in orders_df.iterrows():
        symbol = row['Symbol']
        buy_or_sell = row['Order']
        if buy_or_sell.lower() == 'hold':
            continue

        units = row['Shares']
        units = units if (buy_or_sell == 'BUY') else units * -1
        price = trades_df.ix[index, symbol]
        cash_units = units * price * -1
        trades_df.ix[index, symbol + '_Trade'] += units
        trades_df.ix[index, 'Cash_Trade'] += cash_units

    trades_df.ix[start_dt, 'Cash_Trade'] += start_val
    # Step 3 - Holdings Table
    holdings_df = trades_df.copy()
    holdings_df['Cash_Holdings'] = holdings_df['Cash_Trade'].cumsum(axis=0)
    for symbol in unique_symbols:
        holdings_df[symbol + '_Holdings'] = holdings_df[symbol + '_Trade'].cumsum(axis=0)

    # Step 4- Values Tables
    values_df = holdings_df.copy()
    values_df['Value'] = 0
    for index, row in values_df.iterrows():

        cash_val = values_df.ix[index, 'Cash_Holdings']
        stocks_val_total = 0
        for symbol in unique_symbols:
            price = values_df.ix[index, symbol]
            holding = values_df.ix[index, symbol + '_Holdings']
            values = price * holding
            stocks_val_total += values
        values_df.ix[index, 'Value'] = stocks_val_total + cash_val
        # values_df.to_csv(orders_file+"_result.csv")
    return values_df.ix[:, 'Value']


def calc_order_total(df):
    total = 0
    for index, row in df.iterrows():
        shares = df.ix[index, 'Shares']
        direction = df.ix[index, 'Order']
        if direction.lower() == 'buy':
            total += shares
        elif direction.lower() == 'sell':
            total -= shares
        else:
            total += 0
    return total


def order_list_to_df(order_list):
    columns = ['Date', 'Symbol', 'Order', 'Shares']
    orders_df = pd.DataFrame(order_list, columns=columns)
    orders_df = orders_df.set_index('Date')
    return orders_df


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    unique_symbols = ['AAPL']
    symbol = 'AAPL'
    start_dt = '2008-01-02'
    last_dt = '2009-12-31'
    date_range = pd.date_range(start_dt, last_dt)
    prices_df = get_data(symbols=unique_symbols,
                         dates=date_range,
                         addSPY=True,
                         colname='Adj Close')

    df = calc_bb(prices_df, symbol, 20)
    df = calc_momentum(df, symbol, 20)
    df = calc_std(df, symbol, 20)
    df = calc_ema(df, symbol, 9)
    df = calc_macd(df, symbol, 12, 26)
    df = z_score_df(df)

    start_dt = '2008-01-02'
    end_dt = '2009-12-31'
    rule_orders_df, portvals_rule_normalized = rule_based.get_rule_based(displayPlot=False,
                                                                         start_dt=start_dt,
                                                                         last_dt=end_dt,
                                                                         of_benchmark='./orders/orders_mc3p3_benchmark.csv')
    plt.cla()

    ml_orders_df, predict_df, portvals_ml_normalized, portvals_benchmark_normalized = ml_based.get_ml_based_orders(
        displayPlot=True,
        start_dt=start_dt,
        last_dt=end_dt,
        of_benchmark='./orders/orders_mc3p3_benchmark.csv')

    plt.cla()

    ax1 = portvals_ml_normalized.plot(title="ML Based Portfolio", color='g', label='ML Based Portfolio')
    portvals_rule_normalized.plot(label='Rule Based Portfolio', color='b')
    portvals_benchmark_normalized.plot(label='Benchmark', color='k', ax=ax1)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    ax1.legend(loc='upper left')
    plt.grid(True)


    # prices_normalized = z_score_df(prices_df)
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Price")
    # prices_normalized['AAPL'].plot(color='c', label='Price', ax=ax2)
    # ax2.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    test_code()
