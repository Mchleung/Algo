import numpy as np
import pandas as pd
from asset import StockData
import datetime
import requests
import random
from bs4 import BeautifulSoup

# Temporary
def scrape_constituents(index):
    """
    :param index: Slickcharts have 'nasdaq100', 'sp500', and 'dowjones'
    :return: List of tickers of index constituents
    """
    url = 'https://www.slickcharts.com/{}'.format(index)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find("table", {"class": "table table-hover table-borderless table-sm"})
    table_df = pd.read_html(str(table))
    return list(table_df[0]['Symbol'])


def check_n_days_high(stock_list, start_date, end_date, period=5):
    d = StockData(stock_list, start_date, end_date, 'd')
    n_days_high_dict = {s: d.price_data[s]['adj_close'].iloc[-1] == d.price_data[s]['adj_close'].max() for s in stock_list}
    high_list = [s for s in n_days_high_dict.keys() if n_days_high_dict[s]]
    high_percentage = len(high_list)/len(stock_list)
    print("{}-days New high: {:.2f}% ({}/{})".format(period, high_percentage*100, len(high_list), len(stock_list)))
    return high_list


def check_exchange_n_days_high(exchange, period=5):
    last_day = datetime.datetime.now() - datetime.timedelta(days=30) # US stock price API malfunction
    start_date = last_day - datetime.timedelta(days=period)
    start_date, end_date = start_date.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")
    location = 'us'
    if exchange.lower() == 'sehk':
        location = 'hk'
    canonical_uri = "https://api.invbots.com/data/v1/stock/{}".format(location)
    r = requests.get(canonical_uri)
    df = pd.DataFrame(r.json())
    df = df[df['exchange'].str.lower() == exchange.lower()]
    exchange_stock_list = list(df['ticker'])
    return check_n_days_high(random.choices(exchange_stock_list, k=20), start_date, end_date, period)


def check_index_n_days_high(index, period=5):
    last_day = datetime.datetime.now() - datetime.timedelta(days=30) # US stock price API malfunction
    start_date = last_day - datetime.timedelta(days=period)
    start_date, end_date = start_date.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")
    # TODO: Get real stock list of index
    location = 'us'
    if index.lower() == 'hsi':
        location = 'hk'
    canonical_uri = "https://api.invbots.com/data/v1/stock/{}".format(location)
    r = requests.get(canonical_uri)
    df = pd.DataFrame(r.json())
    exchange_stock_list = list(df['ticker'])
    if index == 'SP500':
        exchange_stock_list = scrape_constituents('sp500')
    elif index == 'DJI':
        exchange_stock_list = scrape_constituents('dowjones')
    elif index == 'NASDAQ':
        exchange_stock_list = scrape_constituents('nasdaq100')
    print(exchange_stock_list)
    return check_n_days_high(exchange_stock_list, start_date, end_date, period)


def check_index_momentum(index):
    print(check_index_n_days_high(index, period=5)) # Weekly
    print("===========================================")
    print(check_index_n_days_high(index, period=21)) # Monthly (Avg. trading days of 21)
    print("===========================================")
    print(check_index_n_days_high(index, period=63)) # Quarterly
    print("===========================================")
    print(check_index_n_days_high(index, period=252)) # Annual


if __name__ == '__main__':
    # s_list = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'V', 'MA', 'BABA']
    # dd = check_n_days_high(s_list, 5)
    # print("Stock reaching 5-day high yesterday:", ', '.join(dd))
    check_index_momentum("DJI")



