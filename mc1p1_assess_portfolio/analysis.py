"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def compute_daily_returns(daily_pos_df):
    """Compute and return the daily return values."""
    daily_returns = daily_pos_df.copy()
    # compute daily returns for row 1 onwards
    df_temp =  daily_pos_df.sum(axis=1)
    daily_returns = (df_temp / df_temp.shift(1)) - 1
    #daily_returns.ix[0, :] = 0  # set daily returns for row 0 to 0
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
    return (port_val[-1] / port_val[0])-1

def compute_sr(daily_returns, rf, sf):
    return ((daily_returns - rf).mean() / daily_returns.std(ddof=1)) * np.sqrt(sf)

def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), \
                     syms=['GOOG', 'AAPL', 'GLD', 'XOM'], \
                     allocs=[0.1, 0.2, 0.3, 0.4], \
                     sv=1000000,
                     rfr=0.0,
                     sf=252.0, \
                     gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    normalized = compute_normalized(prices_all)
    # print normalized
    alloced = compute_alloced(normalized, syms, allocs)

    pos_vals = compute_pos_vals(alloced, sv)
    daily_rets = compute_daily_returns(pos_vals)
    port_val = pos_vals.sum(axis=1)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = [compute_cr(port_val), compute_avg_daily_return(daily_rets), compute_sddr(daily_rets),compute_sr(daily_rets, rfr, sf)]  # add code here to compute stats

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_normalized = compute_normalized(df_temp)
        df_normalized.plot(title='Daily Portfolio Value and SPY')
        plt.show()
        pass

    # Add code here to properly compute end value
    ev = sv

    return cr, adr, sddr, sr, ev


def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010, 06, 1)
    end_date = dt.datetime(2010, 12, 31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd=start_date, ed=end_date, \
                                             syms=symbols, \
                                             allocs=allocations, \
                                             sv=start_val, \
                                             rfr=risk_free_rate,
                                             gen_plot=False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    test_code()
