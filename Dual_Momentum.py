from Indicator import *
from Back_Test import *

def strategy(date, context):
    context.strategy_name = "Dual momentum"
    df = context.data
    # dual momentum rules
    # Start of each month,
    # if Momentum score of SPY > SCZ and momentum SPY > 0 -> long SPY
    # if Momentum score of SCZ > SPY and momentum SCZ > 0 -> long SCZ
    # else long treasury

    target_asset = ""
    try:
        if (context.indicator.momentum_combined.loc[date,"SPY"] > context.indicator.momentum_combined.loc[date,"SCZ"]) and (
                context.indicator.momentum_combined.loc[date,"SPY"]>0):
            if (context.indicator.momentum_combined.loc[date,"SPY"] > context.indicator.momentum_combined.loc[date,"QQQ"]) and (
                context.indicator.momentum_combined.loc[date,"SPY"] > context.indicator.momentum_combined.loc[date,"QQQ"]):
                target_asset = "SPY"
            elif (context.indicator.momentum_combined.loc[date,"DIA"] > context.indicator.momentum_combined.loc[date,"QQQ"]) and (
                context.indicator.momentum_combined.loc[date,"DIA"] > context.indicator.momentum_combined.loc[date,"SPY"]):
                target_asset = "DIA"

            else:
                target_asset = "QQQ"

        elif (context.indicator.momentum_combined.loc[date,"SCZ"] > context.indicator.momentum_combined.loc[date,"SPY"]) and (
                    context.indicator.momentum_combined.loc[date,"SCZ"] > 0
            ):
            target_asset = "SCZ"
        else:
            target_asset = "AGG"

        if context.check_holding():
            # already holding
            if context.position["name"].str.contains(target_asset).any():
                return

            else:
                # sell all position
                for index, row in context.position.iterrows():
                    context.order_position(row["name"], -row["position"], df["Adj Close"].loc[date, row["name"]])
                context.order_by_cash_weight(target_asset,0.99 , df["Adj Close"].loc[date, target_asset])
                return
        else:
            context.order_by_cash_weight(target_asset,0.99, df["Adj Close"].loc[date, target_asset])
            return
    except ValueError:
        pass

if __name__ == '__main__':
    start_time = time.time()
    asset_list = "AGG SPY IXUS TLT SCZ QQQ DIA"

    ytd = datetime.now() - timedelta(days=1)
    ytd.strftime("%Y-%m-%d")

    # variable
    start_date = "2008-01-01"
    stop_date = "2020-12-04"
    cash = 100000000

    # get trading date
    date_df = trading_dates(start_date, stop_date)

    # create context variable
    context = Backtest(cash, date_df[0])
    context.date_df = date_df

    context.load_asset_list(asset_list, "2003-01-01", ytd)
    simple_momentum(context)

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
            strategy(loopdate, context)
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
