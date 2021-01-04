from Back_Test import *


# factor model, or decision tree model for detecting crisis

def prepare_data(context):
    context.external_data()
    context.data["VIX_rtn_10"] = context.data["Adj Close"]["^VIX"].shift(-1) / context.data["Adj Close"][
        "^VIX"].rolling(10).min() - 1
    context.data["VIX_rtn_10_pos"] = context.data["Adj Close"]["^VIX"].shift(-1) / context.data["Adj Close"][
        "^VIX"].rolling(
        10).max() - 1


def strategy(date, df, asset, context):
    # maintain 25% of gold, treasury bond , SPY and cash for rebalance
    context.strategy_name = "Crisis"

    try:
        if (context.indicator.HY.loc[date, "HY_rtn_10"] >= 0.1):
            context.clear_pos()
            # 20% of cash
            context.order_by_weight("TLT", 0.4, df["Adj Close"].loc[date, "TLT"])
            context.order_by_weight("GLD", 0.4, df["Adj Close"].loc[date, "GLD"])
            if context.crisis_empty_period == 0:
                context.crisis_empty_period = 10
            else:
                context.crisis_empty_period += 1
        else:
            if context.crisis_empty_period != 0:
                if context.crisis_empty_period - 1 == 0:
                    context.clear_pos()
                # see if the high yield index drop back, signal market recover
                if context.indicator.HY.loc[date, "HY_rtn_10_pos"] <= -0.1 and float(
                        context.data.loc[date, "VIX_rtn_10_pos"]) >= 0.1:
                    context.crisis_empty_period = 0
                    context.clear_pos()
                    return
                context.crisis_empty_period -= 1

    except KeyError:
        return


if __name__ == '__main__':
    start_time = time.time()

    # variable
    ytd = datetime.now() - timedelta(days=1)
    ytd.strftime("%Y-%m-%d")

    start_date = "2008-01-01"
    stop_date = "2020-06-16"
    cash = 100000000

    # get trading date
    date_df = trading_dates(start_date, stop_date)

    # create context variable
    context = Backtest(cash, date_df[0])
    context.date_df = date_df

    asset_list = "SPY GLD TLT ^VIX ^TNX ^FVX"

    context.load_asset_list(asset_list, "2007-07-02", ytd)
    prepare_data(context)

    # enter daily or monthly for the rebalance period
    rebalance_df = rebalance("daily", date_df)
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
