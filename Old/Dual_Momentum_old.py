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
from Back_Test import *

def prep_data(conext):
    asset_list = "AGG SPY IXUS"

    ytd = datetime.now() - timedelta(days=1)
    ytd.strftime("%Y-%m-%d")
    conext.load_asset_list(asset_list,"2010-01-01",ytd)
    data = yf.download(asset_list, start="2010-01-01", end=ytd)
    data = data["Adj Close"]
    conext.data = data
    return

def strategy(date, df, asset, context):
    context.strategy_name = "dual momentum"

    # one yr momentum
    for i in range(365, 370):
        if ((datetime.strptime(date, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')) in df.index:
            search_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')
            break
    list = []
    for i in range(0, len(asset)):
        list.append(
            (asset[i], df.loc[date, asset[i]] / df.loc[search_date, asset[i]] - 1))

    # dual momentum rules
    # Start of each month,
    # if SPY > Ex US and SPY > AGG -> SPY
    # if SPY > Ex US and SPY < AGG -> AGG
    # if SPY < Ex US and Ex US > AGG -> EX US
    # if SPY < Ex US and Ex US < AGG -> AGG

    # 0:AGG
    # 1:SPY
    # 2:IXUS

    if list[1][1] > list[2][1]:
        if list[1][1] > list[0][1]:
            target_asset = "SPY"
        else:
            target_asset = "AGG"
    else:
        if list[2][1] > list[0][1]:
            target_asset = "IXUS"
        else:
            target_asset = "AGG"

    if context.check_holding():
        # already holding
        if context.position["name"].str.contains(target_asset).any():
            return

        else:
            # sell all position
            for index, row in context.position.iterrows():
                context.order_position(row["name"], -row["position"], df.loc[date, row["name"]])

                # context.position.loc[0, "name"])]
            context.order_position(target_asset,
                                   int(context.cash / df.loc[
                                       date, target_asset]),
                                   df.loc[date, target_asset]
                                   )
            return
    else:
        context.order_position(target_asset, int(context.cash / df.loc[date, target_asset]), df.loc[date, target_asset])
        return

if __name__ == '__main__':
    start_time = time.time()
    asset_list = ["AGG", "SPY", "IXUS"]
    # variable
    start_date = "2014-01-01"
    stop_date = "2020-12-04"
    cash = 100000000

    # get trading date
    date_df = trading_dates(start_date, stop_date)

    # create context variable
    context = Backtest(cash, date_df[0])
    context.date_df = date_df

    prep_data(context)
    # enter daily or monthly for the rebalance period
    rebalance_df = rebalance("monthly", date_df)
    # change date format
    rebalance_df = [x.strftime('%Y-%m-%d') for x in rebalance_df]

    # Start rebalance
    # loop through every trading date
    for loopdate in date_df:
        context.update_date(loopdate)  # update date
        loopdate = loopdate.strftime('%Y-%m-%d')
        context.benchmark_update("SPY")  # for the benchmark
        context.update_nav()  # update nav

        if loopdate in rebalance_df:
            strategy(loopdate, context.data, asset_list, context)
    pd.set_option('display.max_rows', None)

    # plot the nav curve
    context.update_return()

    # get the risk free rate table
    t_rate = get_interest_rate()
    # select treasury duration based on trading period
    rf = t_rate.loc[stop_date, treasury_duration(start_date, stop_date)] / 100
    context.risk.risk_metric(rf, context)

    context.plot()
    # print(context.record)
    # print(context.benchmark)
    # print(context.nav)

    print("Program Run Time %s seconds" % (time.time() - start_time))

