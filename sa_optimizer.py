import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spo


def optimize(target, start, period_n, monthly_return, monthly_fee, timing_of_payment):
    bounds = [(0,target)]
    x0 = target / period_n
    result = spo.minimize(error, x0, args=(target, start, period_n, monthly_return, monthly_fee,timing_of_payment,),
                          method='SLSQP',
                          options={'disp':True, 'eps':1,'maxiter':100},
                          bounds=bounds
                        )
    return result.x


def error(repayment, target, start, period_n, monthly_ret, monthly_fee, timing_of_payment):
    fv = np.fv(monthly_ret - monthly_fee, period_n, -repayment, -start, when=timing_of_payment)
    diff = target - fv[0]
    return abs(diff)


# e.g. I start with sum `1000`,
# and my initial estimated amount to be added each month is 200. At the end of period 1,
# I have `(1000 - 1) * (1 + monthlyRet) + monthlyPymt` (0.1% fees)
def test_code():
    timing_of_payment = 'begin'
    #timing_of_payment = 'end'
    start = 1000
    target = 8669
    monthly_fee = 0.001
    monthly_return = 0.03
    repayment = 500
    n = 12
    calculated_optimal_repayment = optimize(target, start, n, monthly_return, monthly_fee, timing_of_payment)
    result_fv = np.fv(monthly_return, n, -calculated_optimal_repayment, -start, when=timing_of_payment)
    diff = target - result_fv
    print("Monthly Return Used: " + str(monthly_return*100) +"%")
    print("Monthly Fee Applied: " + str(monthly_fee*100) +"%")
    print("Initial Deposit: " + str(start))
    print("Number of payment periods in months: " + str(n))
    print("Optimized Monthly Repayment Value: " + str(calculated_optimal_repayment))
    print("Future Value Targted: " + str(target))
    print("Future Value Using the Calculated Repayment Value: " + str(result_fv))
    print("Future Value Difference to Target Value:" + str(diff))

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called