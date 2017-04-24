"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut


# Moving Average
def calc_sma(df, n):
    col_name = 'SMA_' + str(n)
    sma = pd.Series(pd.rolling_mean(df['Adj Close'], n), name=col_name)
    df = df.join(sma)
    df['Adj Close / SMA'] = df[col_name] / df['Adj Close']
    return df


# Exponential Moving Average
def calc_ema(df, n):
    EMA = pd.Series(pd.ewma(df['Adj Close'], span=n, min_periods=n - 1), name='EMA')
    df = df.join(EMA)
    return df


# Bollinger Bands
def calc_bb_bands(df, n):
    MA = pd.Series(pd.rolling_mean(df['Adj Close'], n))
    MSD = pd.Series(pd.rolling_std(df['Adj Close'], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='Bollinger_Lower')
    df = df.join(B1)
    b2 = (df['Adj Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger_Upper')
    df = df.join(B2)
    return df


# MACD, MACD Signal and MACD difference
def calc_macd(df, n_fast, n_slow):
    ema_fast = pd.Series(pd.ewma(df['Adj Close'], span=n_fast, min_periods=n_slow - 1))
    ema_slow = pd.Series(pd.ewma(df['Adj Close'], span=n_slow, min_periods=n_slow - 1))
    col_name_macd = 'MACD'
    col_name_sign = 'MACDSign'
    col_name_macd_diff = 'MACDDiff'
    macd = pd.Series(ema_fast - ema_slow, name=col_name_macd)
    macd_sign = pd.Series(pd.ewma(macd, span=9, min_periods=8), name=col_name_sign)
    macd_diff = pd.Series(macd - macd_sign, name=col_name_macd_diff)
    df = df.join(macd)
    df = df.join(macd_sign)
    df = df.join(macd_diff)
    return df


def discretize(df, steps):
    stepsize = df.shape[0] / steps
    sorted_by_macd = df.sort_values(by='MACD', axis=0, ascending=True)
    columns = ['MACD', 'MACDSign', 'MACDDiff', 'Bollinger_Lower', 'Bollinger_Upper', 'EMA', 'Adj Close / SMA']
    discretize_df = pd.DataFrame(index=range(0, 9), columns=columns)
    for i in range(0, steps):
        discretize_df.ix[i, 'MACD'] = sorted_by_macd['MACD'].iloc[(0 + 1) * i]

    sorted_by_macd_sign = df.sort_values(by='MACDSign', axis=0, ascending=True)
    for i in range(0, steps):
        discretize_df.ix[i, 'MACDSign'] = sorted_by_macd_sign['MACDSign'].iloc[(0 + 1) * i]

    sorted_by_macd_diff = df.sort_values(by='MACDDiff', axis=0, ascending=True)
    for i in range(0, steps):
        discretize_df.ix[i, 'MACDDiff'] = sorted_by_macd_diff['MACDDiff'].iloc[(0 + 1) * i]

    sorted_by_bb_lower = df.sort_values(by='Bollinger_Lower', axis=0, ascending=True)
    for i in range(0, steps):
        discretize_df.ix[i, 'Bollinger_Lower'] = sorted_by_bb_lower['Bollinger_Lower'].iloc[(0 + 1) * i]

    sorted_by_bb_upper = df.sort_values(by='Bollinger_Upper', axis=0, ascending=True)
    for i in range(0, steps):
        discretize_df.ix[i, 'Bollinger_Upper'] = sorted_by_bb_upper['Bollinger_Upper'].iloc[(0 + 1) * i]

    sorted_by_ema = df.sort_values(by='EMA', axis=0, ascending=True)
    for i in range(0, steps):
        discretize_df.ix[i, 'EMA'] = sorted_by_ema['EMA'].iloc[(0 + 1) * i]

    sorted_by_ema = df.sort_values(by='Adj Close / SMA', axis=0, ascending=True)
    for i in range(0, steps):
        discretize_df.ix[i, 'Adj Close / SMA'] = sorted_by_ema['Adj Close / SMA'].iloc[(0 + 1) * i]

    return discretize_df


class StrategyLearner(object):
    # constructor
    def __init__(self, verbose=False):
        self.verbose = verbose

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="IBM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 1, 1), \
                    sv=10000):

        self.learner = ql.QLearner()
        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        # example use with new colname 
        volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols

        prices_and_volumes = pd.DataFrame(data=None, index=prices.index, columns=['Adj Close', 'Volume'])
        prices_and_volumes['Adj Close'] = prices[symbol]
        prices_and_volumes['Volume'] = volume[symbol]

        df_with_macd = calc_macd(prices_and_volumes, 12, 26)
        df_with_ema = calc_ema(df_with_macd, 9)
        df_with_bb_bands = calc_bb_bands(df_with_ema, 20)
        df_with_sma = calc_sma(df_with_bb_bands, 5).fillna(0)
        df_discretized = discretize(df_with_sma, 9)
        print df_discretized

        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol, ]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:, :] = 0  # set them all to nothing
        trades.values[3, :] = 500  # add a BUY at the 4th date
        trades.values[5, :] = -500  # add a SELL at the 6th date
        trades.values[6, :] = -500  # add a SELL at the 7th date
        trades.values[8, :] = 1000  # add a BUY at the 9th date
        if self.verbose: print type(trades)  # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades


if __name__ == "__main__":
    print "One does not simply think up a strategy"
