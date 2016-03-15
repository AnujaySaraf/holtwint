# import needed packages
#-----------------------

import numpy  as np
import pandas as pd

from sklearn        import linear_model
from scipy.optimize import fmin_l_bfgs_b

pd.set_option('display.max_rows', 500)
pd.set_option('display.width',    100)

# bring in the test data set from homework 4
#-------------------------------------------

sdata = open('sampledata.csv')
tsA = sdata.read().split('\n')
tsA = list(map(int, tsA))

cdata = open('HourCount.csv')
tsB = [line.split(",")[1].strip("\n") for line in cdata.readlines()]
tsB = list(map(int, tsB[1:]))

# define functions to find initial values of (a,b,s) and calculate HoltWinters predictions
#-----------------------------------------------------------------------------------------

def holtWinters(ts, p, sp, ahead, alpha, beta, gamma):
    '''additive HoltWinters retrospective smoothing & prediction algorithm
    @ts[list]:     time series of data to model
    @p[int]:       period of the time series (for calculation of seasonal effects)
    @sp[int]:      number of starting periods to use when calculating initial parameter values
    @ahead[int]:   number of future periods for which predictions will be generated
    @alpha[float]: forgetting factor [0,1] for the level component in smoothing/prediction
    @beta[float]:  forgetting factor [0,1] for the slope component in smoothing/prediction
    @gamma[float]: forgetting factor [0,1] for the seasonality component in smoothing/prediction
    @return: 
        - @params[tuple]:   final (a,b,s) parameter values
        - @MSD[float]:      Mean Square Deviation of one-step-ahead predictions on the original time series
        - @smoothed[list]:  smoothed values (Level + Trend + Seasonal) for the original time series
        - @predicted[list]: predicted values for the next @ahead periods in the time series
    sample call:
        results = holtWinters(ts, 12, 4, 24, 0.1, 0.1, 0.1)'''

    ivals = _initValues(ts, p, sp)
    MSD, smoothed, params = _expSmooth(ts, p, ivals, alpha, beta, gamma)
    predicted = _predictValues(p, ahead, params)
    return {'params': params, 'MSD': MSD, 'smoothed': smoothed, 'predicted': predicted}

def _initValues(ts, p, sp):
    '''subroutine to calculate the initial parameter values (a, b, s) for the HoltWinters algorithm'''

    initSeries = pd.Series(ts[:p*sp])
    rawSeason  = initSeries - pd.rolling_mean(initSeries, window = p, min_periods = p, center = True)
    initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
    initSeason = pd.Series(initSeason) - np.mean(initSeason)
    deSeasoned = [initSeries[v] - initSeason[v % p] for v in range(len(initSeries))]

    lm = linear_model.LinearRegression()
    lm.fit(pd.DataFrame({'time': [t+1 for t in range(len(initSeries))]}), pd.Series(deSeasoned))
    return float(lm.intercept_), float(lm.coef_), list(initSeason)

def _expSmooth(ts, p, ivals, alpha, beta, gamma):
    '''subroutine to calculate the retrospective smoothed values, final parameter values for prediction, and MSD'''

    smoothed = []
    Lt1, Tt1, St1 = ivals[:]

    for t in range(len(ts)):

        Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
        Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
        St = gamma * (ts[t] - Lt)         + (1 - gamma) * (St1[t % p])

        smoothed.append(Lt1 + Tt1 + St1[t % p])
        Lt1, Tt1, St1[t % p] = Lt, Tt, St

    MSD = sum([(ts[t] - smoothed[t])**2 for t in range(len(smoothed))])/len(smoothed)
    return MSD, smoothed, (Lt1, Tt1, St1)

def _predictValues(p, ahead, params):
    '''subroutine to generate predicted values @ahead periods into the future'''

    Lt, Tt, St = params
    return [Lt + (t+1)*Tt + St[t % p] for t in range(ahead)]

# print out the results to check against the R output
#----------------------------------------------------

results = holtWinters(tsB, 168, 4, 168, 0.1, 0.0, 0.1)

print("MSD", results['MSD'])
print("PARAMS", results['params'])
print("PREDICTED", results['predicted'])






