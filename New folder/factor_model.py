import numpy as np
import statsmodels.api as sm
from statsmodels import regression
import treasury_rate as tr
import util.data_access as da
import pandas as pd
import math
import datetime


def lin_reg(x, y: np.ndarray):
    """
    Linear Regression from statsmodel
    :type x: (Array of) numpy.ndarray
    :type y: numpy.ndarray
    :param x: Series of data
    :param y: Series of data
    :return: Alpha, Beta and Residual Error of OLS regression
    """
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    x = x[:, 1]
    print(model.summary())
    return model.params[0], model.params[1], float(np.std(model.resid))  # Alpha, Beta, Residual Error


def build_factor_model(ticker, start_date, end_date, location, period, rates=[], misc=[], window_width=90,
                       price_df=pd.DataFrame()):
    """

    :param ticker: Target asset
    :param start_date: '%Y-%m-%d' format str
    :param end_date: '%Y-%m-%d' format str
    :param location: 'hk' or 'us'
    :param period: '1d' or '1w' or '1mo' or '1q' (for quarterly) or '1y'
    :param rates: rate factor from ('3 mo', '6 mo', '1 yr', '2 yr', '3 yr', ..., '20 yr', '30 yr')
    :param misc: other factors
    :param price_df: Price of ticker if provided
    :return: Tuple of params of regression on all data, and rolling regression params dict on particular date
    """
    print(window_width)
    if price_df.empty:
        price_df = da.get_stock_price_from_data_api(ticker, start_date, end_date, location)
        if price_df.empty:
            price_df = tr.get_index_price_data(ticker, period)
            price_df = price_df.loc[(price_df['date_time'] >= start_date) & (price_df['date_time'] <= end_date)]
    func_dict = {
        '1d': tr.get_interest_rate,
        '1w': tr.weekly_rate,
        '1mo': tr.monthly_rate,
        '1q': tr.quarterly_rate,
        '1y': tr.yearly_rate
    }
    assert period in func_dict.keys(), 'Wrong period type'
    rate_df = func_dict[period]()
    rate_df = rate_df.loc[(rate_df['date_time'] >= start_date) & (rate_df['date_time'] <= end_date)]
    # Find intersecting date row
    price_df = price_df.set_index('date_time')
    price_df.index = price_df.index.values.astype(str)
    rate_df = rate_df.set_index('date_time')
    rate_df.index = rate_df.index.values.astype(str)
    price_df = price_df[price_df.index.isin(rate_df.index)]
    rate_df = rate_df[rate_df.index.isin(price_df.index)]
    df_dict = {'close': price_df['adj_close'] if 'adj_close' in price_df.columns else price_df['close']}
    for i in range(len(rates)):
        if rates[i][-1] == 'm':
            rates[i] = rates[i][:-1] + ' ' + 'mo'
        elif rates[i][-1] == 'y':
            rates[i] = rates[i][:-1] + ' ' + 'yr'
        df_dict[rates[i]] = rate_df[rates[i]]
    # print(rates)
    for x in misc:
        df_dict[x] = x
    reg_df = pd.DataFrame(df_dict)
    # print(reg_df)
    # print(len(price_df['close']))
    X = reg_df[rates]
    Y = reg_df[['close']]
    X = sm.add_constant(X)
    # if 'adj_close' in price_df.columns:
    #     model = sm.OLS(price_df['adj_close'], X).fit()
    # else:
    #     model = sm.OLS(price_df['close'], X).fit()
    params_dict = dict()
    last_param = None
    for row in range(window_width, len(reg_df)):
        # print(reg_df.index[row])
        model = sm.OLS(Y.iloc[row - window_width: row], X.iloc[row - window_width: row], missing='drop').fit()
        params_dict[(row, reg_df.index[row])] = model.params.tolist()
        last_param = model.params
    model = sm.OLS(Y, X, missing='drop').fit()
    # print(model.summary())
    # print(model.params.tolist())
    # print(params_dict)
    error_terms = dict()
    n = len(params_dict)
    for k, v in params_dict.items():
        index, date = k
        if 0 < index - window_width < n - 2:
            data = reg_df.iloc[index + 1].tolist()
            predicted = sum([x * y for x, y in zip(v[1:], data[1:])]) + v[0]
            actual = reg_df.iloc[index + 2, 0]
            error_terms[(index, date)] = predicted - actual
    mse = (sum([0 if math.isnan(x ** 2) else x ** 2 for x in list(error_terms.values())]) /
           len([x for x in error_terms.values() if not math.isnan(x)]))
    # print("Root mean square error", mse**0.5)
    return model.params, last_param, mse ** 0.5


def rolling_window_testing(ticker, start_date, end_date, location, period, rates=[], misc=[], lower=5, upper=365,
                           step=1):
    price_df = da.get_stock_price_from_data_api(ticker, start_date, end_date, location)
    if price_df.empty:
        price_df = tr.get_index_price_data(ticker, period)
        price_df = price_df.loc[(price_df['date_time'] >= start_date) & (price_df['date_time'] <= end_date)]
    res = {
        r: build_factor_model(ticker, start_date, end_date, location, period, rates, window_width=r, price_df=price_df)
        for r in range(lower, upper, step)}
    o_width = min(res, key=lambda k: res[k][2])
    print("Testing done: Minimum RMSE obtained for window width: {} with RMSE {}".format(o_width, res[o_width][2]))
    print("Param to be used:", res[o_width][1])
    return res


def predict(ticker, start_date='1996-01-01', end_date=datetime.datetime.now().strftime('%Y-%m-%d'), location='us',
            period='1d', rates=[], misc=[], window_width=10):
    fm = build_factor_model(ticker, start_date, end_date, location, period, rates, misc, window_width)
    predict_param = fm[1].tolist()
    r = tr.get_interest_rate()
    for i in range(len(rates)):
        if rates[i][-1] == 'm':
            rates[i] = rates[i][:-1] + ' ' + 'mo'
        elif rates[i][-1] == 'y':
            rates[i] = rates[i][:-1] + ' ' + 'yr'
    data = [r[rate].iloc[-1] for rate in rates]
    print(data)
    print(predict_param)
    predicted = sum([x * y for x, y in zip(predict_param[1:], data[0:])]) + predict_param[0]
    return predicted


if __name__ == "__main__":
    index = '^RMZ'
    s_date = '2009-01-01'
    e_date = datetime.datetime.now().strftime('%Y-%m-%d')
    loc = 'us'
    period_type = '1d'
    window = 10
    # p = build_factor_model(index, s_date, e_date, loc, period_type, rates=['20y', '30y'], window_width=window)
    # d = rolling_window_testing(index, s_date, e_date, loc, period_type, rates=['20y', '30y'], lower=5, upper=30, step=1)
    # print(p)
    p_value = predict(index, s_date, e_date, loc, period_type, rates=['20y', '30y'], window_width=window)
    print(p_value)
