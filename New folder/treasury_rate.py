import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
from os.path import isfile, getmtime
import matplotlib.pyplot as plt


# Temporary
def get_interest_rate():
    """
    If the rate csv is updated today, read the csv, else scrape from us gov and save as csv
    :return: Treasury rate dataframe
    """
    m_date = datetime.datetime.fromtimestamp(getmtime('interest_rate.csv')).strftime('%Y-%m-%d')
    if isfile('interest_rate.csv') and m_date == datetime.datetime.now().strftime('%Y-%m-%d'):
        return pd.read_csv("interest_rate.csv")

    url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldAll'
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find("table", {"class": "t-chart"})
    table_df = pd.read_html(str(table))[0]
    table_df['Date'] = table_df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d'))
    table_df = table_df.rename(columns={"Date": "date_time"})
    long_rate = get_long_rate()
    fill_dict = long_rate.set_index('date_time')['LT COMPOSITE (>10 yrs)'].to_dict()
    table_df['30 yr'] = table_df['30 yr'].fillna(
        table_df['date_time'].map(fill_dict))  # Fill missing 30 yr from 2002 to 2006
    table_df = table_df.fillna(method='ffill')
    interest_rate_to_csv(table_df)
    return table_df


def get_long_rate():
    """
    Helper function to fill missing 30 yr data from 2002 to 2006
    :return: Long rate dataframe
    """
    url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=longtermrateAll'
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find("table", {"class": "t-chart"})
    table_df = pd.read_html(str(table))[0]
    table_df['DATE'] = table_df['DATE'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d'))
    table_df = table_df.rename(columns={"DATE": "date_time"})
    return table_df


def interest_rate_to_csv(table_df=pd.DataFrame()):
    if table_df.empty:
        get_interest_rate().to_csv('interest_rate.csv', index=False)
    else:
        table_df.to_csv('interest_rate.csv', index=False)


def monthly_rate(rate_df=pd.DataFrame()):
    if rate_df.empty:
        rate_df = get_interest_rate()
    rate_df = rate_df.groupby(pd.DatetimeIndex(rate_df.date_time).to_period('M')).nth(0)
    del rate_df['date_time']
    rate_df = rate_df.reset_index()
    return rate_df


def yearly_rate(rate_df=pd.DataFrame()):
    if rate_df.empty:
        rate_df = get_interest_rate()
    rate_df = rate_df.groupby(pd.DatetimeIndex(rate_df.date_time).to_period('Y')).nth(0)
    rate_df = rate_df.reset_index(drop=True)
    return rate_df


def weekly_rate(rate_df=pd.DataFrame()):
    if rate_df.empty:
        rate_df = get_interest_rate()
    rate_df = rate_df.iloc[::5]
    return rate_df


def quarterly_rate(rate_df=pd.DataFrame()):
    if rate_df.empty:
        rate_df = get_interest_rate()
    rate_df = rate_df.iloc[::63]
    return rate_df


def get_index_price_data(index, period='1mo'):
    """

    :param index:
    :param period: '1mo' or '1d'
    :return: Price data
    """
    host = 'query1.finance.yahoo.com/'
    canonical_uri = 'v8/finance/chart/{}'.format(index)
    endpoint = "https://" + host + canonical_uri
    r = requests.get(endpoint, params={'interval': period, 'range': '30y'})
    print(r.url)
    print(r.status_code)
    timestamp = r.json()['chart']['result'][0]['timestamp']
    df = pd.DataFrame(r.json()['chart']['result'][0]['indicators']['quote'][0])
    format_string = '%Y-%m' if period == '1mo' else '%Y-%m-%d'
    df['date_time'] = timestamp
    df['date_time'] = df.date_time.apply(lambda x: pd.datetime.fromtimestamp(x).strftime(format_string))
    df.set_index(df['date_time'])
    print(df)
    return df


def build_monthly_corr(index, rolling_window=36):
    """
    :param index: Index ticker
    :param rolling_window: Rolling window in month
    :return: DataFrame of rolling correlation of monthly data on index close and treasury yield
    """
    monthly = monthly_rate()
    index_price_data = get_index_price_data(index, '1mo')[:-1]
    del monthly['1 mo']
    del monthly['2 mo']
    monthly = monthly[monthly['date_time'] >= '1995-06-01']
    monthly = monthly.set_index('date_time')
    monthly.index = monthly.index.values.astype(str)
    index_price_data = index_price_data[index_price_data['date_time'] >= '1995-06-01']
    index_price_data = index_price_data[['date_time', 'close']]
    index_price_data = index_price_data.set_index('date_time')
    index_price_data.index = index_price_data.index.values.astype(str)
    print(index_price_data.rolling(rolling_window).corr(monthly['3 mo']))
    corr_df = index_price_data.rolling(rolling_window).corr(monthly['3 mo'])
    corr_df = corr_df.rename(columns={'close': '3 mo'})
    print(corr_df)
    for r in monthly.columns:
        if r != '3 mo':
            corr_df[r] = index_price_data.rolling(rolling_window).corr(monthly[r])['close']
    corr_df = corr_df.dropna()
    return corr_df


def build_daily_corr(index, rolling_window=2520):
    """
    :param index: Index ticker
    :param rolling_window: Rolling window in month
    :return: DataFrame of rolling correlation of monthly data on index close and treasury yield
    """
    daily = get_interest_rate()
    index_price_data = get_index_price_data(index, '1d')
    index_price_data = index_price_data.fillna(method='ffill')
    del daily['1 mo']
    del daily['2 mo']
    daily = daily[daily['date_time'] >= '1995-07-01']
    daily = daily.set_index('date_time')
    daily.index = daily.index.values.astype(str)
    print(daily)
    index_price_data = index_price_data[index_price_data['date_time'] >= '1995-07-01']
    index_price_data = index_price_data[['date_time', 'close']]
    index_price_data = index_price_data.set_index('date_time')
    index_price_data.index = index_price_data.index.values.astype(str)

    index_price_data = index_price_data[index_price_data.index.isin(daily.index)]
    daily = daily[daily.index.isin(index_price_data.index)]

    print("len", len(daily), len(index_price_data))
    print(index_price_data.rolling(rolling_window).corr(daily['3 mo']))
    corr_df = index_price_data.rolling(rolling_window).corr(daily['3 mo'])
    corr_df = corr_df.rename(columns={'close': '3 mo'})
    for r in daily.columns:
        if r != '3 mo':
            corr_df[r] = index_price_data.rolling(rolling_window).corr(daily[r])['close']
    corr_df = corr_df.dropna()
    return corr_df


if __name__ == '__main__':
    # print(get_interest_rate().tail())
    # print(monthly_rate(rate).tail())
    # print(weekly_rate(rate).tail())
    # print(quarterly_rate(rate).tail())
    # print(yearly_rate(rate).tail())

    rolling_day = 252 * 10
    rolling_month = 12 * 20
    target_index = "^RMZ"
    # monthly_corr = build_monthly_corr(target_index, rolling_window=rolling)
    # print(monthly_corr)
    # monthly_corr.plot(figsize=(12, 9), title='{}-month rolling correlation of {} with treasury yield'.format(rolling, target_index))
    # plt.savefig('./fig/{}-mo-rolling-corr-{}-w-yield.png'.format(rolling, target_index))
    daily_corr = build_daily_corr(target_index, rolling_window=rolling_day)
    monthly_corr = build_monthly_corr(target_index, rolling_window=rolling_month)
    print(daily_corr)
    print(monthly_corr)
    daily_corr.plot(figsize=(12, 9),
                    title='{}-day rolling correlation of {} with treasury yield'.format(rolling_day, target_index))
    monthly_corr.plot(figsize=(12, 9),
                      title='{}-month rolling correlation of {} with treasury yield'.format(rolling_month,
                                                                                            target_index))
    plt.show()
