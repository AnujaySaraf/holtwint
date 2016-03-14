# import needed packages
#-----------------------

import numpy  as np
import pandas as pd

from sklearn        import linear_model
from scipy.optimize import fmin_l_bfgs_b

# bring in the test data set from homework 4
#-------------------------------------------

sdata = open('sampledata.csv')
slist = sdata.read().split('\n')
slist = list(map(int, slist))

# define functions to find initial values of (a,b,s) and calculate HoltWinters predictions
#-----------------------------------------------------------------------------------------

def initValues(slist, period, nperiods):

    initSeries = pd.Series(slist[:period*nperiods])
    rawSeason  = initSeries - pd.rolling_mean(initSeries, window = period, min_periods = period, center = True)
    initSeason = [np.nanmean(rawSeason[i::period]) for i in range(period)]
    initSeason = pd.Series(initSeason) - np.mean(initSeason)
    deSeasoned = [initSeries[v] - initSeason[v % period] for v in range(len(initSeries))]

    lm = linear_model.LinearRegression()
    lm.fit(pd.DataFrame({'time': [t+1 for t in range(len(initSeries))]}), pd.Series(deSeasoned))
    return float(lm.intercept_), float(lm.coef_[0]), list(initSeason)


def holtWinters(slist, period, ahead, a0, b0, s0, alpha, beta, gamma):

    smoothed = []
    Lt1, Tt1, St1 = a0, b0, s0

    for i in range(len(slist)):

        Lt = alpha * (slist[i] - St1[i % period]) + (1 - alpha) * (Lt1 + Tt1)
        Tt = beta  * (Lt - Lt1)                   + (1 - beta)  * (Tt1)
        St = gamma * (slist[i] - Lt)              + (1 - gamma) * (St1[i % period])

        smoothed.append(Lt)
        Lt1, Tt1, St1[i % period] = Lt, Tt, St

    predictions = [Lt1 + (t+1)*Tt1 + St1[t % period] for t in range(ahead)]
    return smoothed, predictions


def MSD(params, *args):

    alpha, beta, gamma = params
    slist, period      = args[0], args[1]
    Lt1, Tt1, St1      = args[2], args[3], args[4] 

    forecasts = []

    for i in range(len(slist)):

        Lt = alpha * (slist[i] - St1[i % period]) + (1 - alpha) * (Lt1 + Tt1)
        Tt = beta  * (Lt - Lt1)                   + (1 - beta)  * (Tt1)
        St = gamma * (slist[i] - Lt)              + (1 - gamma) * (St1[i % period])

        forecasts.append(Lt1 + Tt1 + St1[i % period])
        Lt1, Tt1, St1[i % period] = Lt, Tt, St

    return sum([(slist[t]-forecasts[t])**2 for t in range(len(slist))])/len(forecasts)

# try to get the optimization functions to find the best (alpha, beta, gamma) values
#-----------------------------------------------------------------------------------

initial  = initValues(slist, 12, 4)
smoothed = holtWinters(slist, 12, 24, initial[0], initial[1], initial[2], 0.1, 0.1, 0.1)

init   = [0.1, 0.1, 0.1]
bounds = [(0, 1), (0, 1), (0, 1)]

parameters = fmin_l_bfgs_b(MSD, x0 = init, args = (slist, 12, initial[0], initial[1], initial[2]), bounds = bounds, approx_grad = True)

alpha, beta, gamma = parameters[0]
finalMSD           = parameters[1]
messages           = parameters[2]

print(list(map(round, smoothed[1])))
print(alpha, beta, gamma)
print(finalMSD)
print(messages)



