import quandl
import datetime
import pandas as pd
from xone import calendar
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly as py
# import monthly_returns_heatmap as mrh
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import math
from Back_Test import *

def simple_moving_average(context):
    for ticker in context.asset_list:
        context.indicator.SMA[ticker + "SMA100"] = context.data["Adj Close"][ticker].rolling(100).mean()
        context.indicator.SMA[ticker + "SMA150"] = context.data["Adj Close"][ticker].rolling(100).mean()
        context.indicator.SMA[ticker + "SMA200"] = context.data["Adj Close"][ticker].rolling(200).mean()

def simple_momentum(context):
    context.indicator.ultra_short_momentum = context.data["Adj Close"].pct_change(21)
    context.indicator.short_momentum = context.data["Adj Close"].pct_change(21*3)
    context.indicator.long_momentum = context.data["Adj Close"].pct_change(21*6)
    context.indicator.momentum_combined = (context.indicator.long_momentum + context.indicator.short_momentum + context.indicator.ultra_short_momentum)/3
    return

def momentum(df):
    returns = np.log(df)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = stats.linregress(x, returns)
    return ((1 + slope) ** 252) * (rvalue ** 2)

def averate_true_value(df,ticker,t):
    start_time = time.time()
    atr = pd.DataFrame()
    df = df.reset_index()
    first = True

    # for index, row in df.iterrows():
    # # do some logic here

    for index, row  in  df["Adj Close"].iterrows():
        if index == 0:
            atr.loc[index, ticker + "TR"] = df["High"].loc[index, ticker] - df["Low"].loc[index, ticker]
            continue

        if not math.isnan(df["Adj Close"].loc[index, ticker]):
            atr.loc[index, ticker+"TR"] = max(df["High"].loc[index,ticker] - df["Low"].loc[index,ticker],
            abs(df["High"].loc[index,ticker] - df["Adj Close"].loc[index-1,ticker]),
            abs(df["Low"].loc[index,ticker] - df["Adj Close"].loc[index-1,ticker]))
        else:
            atr.loc[index, ticker + "TR"]= 0
            continue
    # atr[t-1,ticker+"ATR"] = atr[0:t-1,ticker + "TR"].mean()
    # first_atr = atr.loc[0,ticker+"ATR"]
    # atr[0, ticker + "ATR"] = first_atr
    first = True
    for i in range(0,len(df["Adj Close"][ticker])):
        if not math.isnan(df["Adj Close"].loc[i,ticker]):
            if first:
                atr.loc[i, ticker + "ATR"] = df["High"].loc[i, ticker] - df["Low"].loc[i, ticker]
                first = False
            else:
                atr.loc[i, ticker + "ATR"] = (((atr.loc[i - 1, ticker + "ATR"] * (t - 1)) + atr.loc[i, ticker + "TR"]) / t)
        else:
            atr.loc[i, ticker + "ATR"] = 0
            continue

    atr = atr.sort_index()
    atr = atr[ticker+"ATR"].tolist()
    # print(ticker + (" ATR run Time %s seconds" % (time.time() - start_time)))

    return atr # return the ATR list

# if __name__ == '__main__':
#

