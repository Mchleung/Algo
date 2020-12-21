from Back_Test import *
from Indicator import *
from SPX_Historical_Constituents import *

### Created by Leung Cheuk Hang Matthew

#TODO: Position Dod with unrealized pnl

#TODO: list import from excel of stock selecting

def volatility(context,t):
    volList = context.data["Adj Close"].rolling(t,min_periods=1).std(ddof=0)
    return volList

# def SMA(df,time):
def prepare_data(context):
    df = context.data
    context.momentums = pd.DataFrame()
    # momentums = context.data.copy(deep=True)
    indic = pd.DataFrame()
    #
    context.volatility = volatility(context, 180)
    context.inverse_volatility = 1 / context.volatility
    for ticker in context.asset_list:
        context.momentums[ticker] = context.data["Adj Close"][ticker].rolling(90).apply(momentum, raw=False)

    simple_moving_average(context)

    # ## context.indic[ticker + "ATR"] = averate_true_value(df,ticker,20)

        # print(" Run Time %s seconds for" % (time.time() - start_time))
    context.rankings = context.momentums.rank(1, ascending=False,numeric_only=True,na_option="bottom")

def strategy(date, context):
    context.strategy_name = "S&P Momentum"
    order = []
    total_vol = 0 # trade securities inverse volatility sum
    # rebalance
    if context.check_holding():  # if have holding then

    # Sell position that underperform
        for x in context.position["name"]:
            if (context.rankings.loc[date,x] > 0.2*len(context.rankings.columns)) or (context.data["Open"].loc[date,x]< context.indicator.SMA.loc[date, x + "SMA100"]):
                context.order_position(x,-float(context.position[context.position["name"]==x]["position"]),context.data["Adj Close"].loc[date,x])

    # check if S&P 500 is above 200 SMA if not then do not add new position
    if context.indicator.SMA.loc[date, "SPYSMA200"] > context.data["Adj Close"].loc[date, "SPY"]:
        return
    else:
        i = 0
        k = math.floor(0.2 * len(context.rankings.columns))+1
        while i < k:
            for x in context.asset_list:
                    if context.rankings.loc[date, x] == i:
                        if x in context.SPY_con.loc[context.SPY_con.truncate(after=date).index[-1], "Holdings"]: #check if the position is in S&P500 or not
                            if not math.isnan(context.data["Adj Close"].loc[date, x]):
                                order.append(x)
                            else:
                                k = k + 1
            i = i + 1
        print(order)
        for x in order:
            total_vol += context.inverse_volatility.loc[date,x]
        for x in order: # if x in context.position: then need to adjust the weighting
            if x in context.position["name"]:
                context.order_position(x, math.floor(
                    context.cash * 0.001 * context.inverse_volatility.loc[date, x] / total_vol)-context.position["name"==x]["position"]
                                       ,context.data["Close"].loc[date, x])
            else:
                context.order_by_weight(x,context.inverse_volatility.loc[date, x] / total_vol,context.data["Adj Close"].loc[date, x])

    # for x in context.asset_list: #Testing purpose
    #     context.order_by_cash_weight(x,0.1,context.data["Adj Close"].loc[date, x])

    return

if __name__ == '__main__':
    start_time = time.time()
    # asset_list = "AGG SPY IXUS AAPL AMZN TSLA WFC JPM" #Testing Dataset

    ytd = datetime.now() - timedelta(days=1)
    ytd.strftime("%Y-%m-%d")

    # variable
    start_date = "2018-01-01"
    stop_date = "2020-12-11"
    cash = 100000000 # 100M USD

    # get trading date
    date_df = trading_dates(start_date, stop_date)

    # create context variable
    context = Backtest(cash, date_df[0])
    context.date_df = date_df

    asset_list = "SPY AGG "
    df, context.SPY_con = process_constituents()
    # print(context.SPY_con)
    context.SPY_con.index = context.SPY_con["Date"]
    for ticker in df:
        asset_list = asset_list + ticker + " "
    asset_list = asset_list.strip()

    context.load_asset_list(asset_list, "2015-01-01", ytd)

    # enter daily or monthly for the rebalance period
    rebalance_df = rebalance("weekly", date_df)
    # change date format
    rebalance_df = [x.strftime('%Y-%m-%d') for x in rebalance_df]
    prepare_data(context)

    # Start rebalance
    # loop through every trading date
    for loopdate in date_df:
        context.update_date(loopdate)  # update date
        loopdate = loopdate.strftime('%Y-%m-%d')
        context.benchmark_update("SPY")  # for the benchmark

        if loopdate in rebalance_df:
            strategy(loopdate, context)
        context.update_nav()  # update nav

    pd.set_option('display.max_rows', None)

    # plot the nav curve
    context.update_return()

    # get the risk free rate table
    t_rate = get_interest_rate()
    # select treasury duration based on trading period
    # rf = t_rate.loc[stop_date, treasury_duration(start_date, stop_date)] / 100
    context.risk.risk_metric(0, context)

    context.plot()
    # print(context.record)
    # print(context.benchmark)
    # print(context.nav)

    print("Program Run Time %s seconds" % (time.time() - start_time))



