from util.data_access import get_stock_price_from_data_api
import pandas as pd
import numpy as np

class StockData:
    def __init__(self, stock_list, start_date, end_date, period='d'):
        """
        Contain stock data used by optimization process
        :param stock_list: List of ticker
        :param start_date:
        :param end_date:
        :param period: The interval for stock data. {'d'/'w'/'m'}
        """
        self.stock_list = stock_list
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = {}
        self._corr = pd.DataFrame()
        self._std = {}
        self._return = {}
        self._avg_return = {}

        temp = []
        for s in stock_list:
            if s == "STCGUSRE" or s == "DJR":
                location = "index"
            elif s.isdigit():
                location = "hk"
            else:
                location = "us"
            # print("Getting %s"%s)
            self.price_data[s] = get_stock_price_from_data_api(s, start_date, end_date, location)
            self.price_data[s]['ticker'] = s
            if period == 'm':
                self.price_data[s] = self.price_data[s].groupby(pd.DatetimeIndex(self.price_data[s].date_time).to_period('M')).nth(0)
            elif period == 'w':
                self.price_data[s] = self.price_data[s].iloc[::5]
            self.price_data[s]['change'] = self.price_data[s]['adj_close'].pct_change()
            temp.append(self.price_data[s])

        df = pd.concat(temp)
        # df = df.reset_index()
        df = df[['date_time', 'ticker', 'change']]
        df_pivot = df.pivot('date_time', 'ticker', 'change').reset_index()
        self._corr = df_pivot.corr(method='pearson')
        self.cov = df_pivot.cov()
        self._corr.head().reset_index()
        self.cov.head().reset_index()
        del self._corr.index.name
        del self.cov.index.name

        self.pct_change = {}
        self.pct = pd.DataFrame()
        for stock in self.price_data:
            self.pct_change[stock] = self.price_data[stock]['adj_close'].pct_change()
            self.pct[stock] = self.pct_change[stock]
        self.pct.dropna(how='any')
        for stock in self.price_data:
            self._return[stock] = self.pct_change[stock].sum()
            self._avg_return[stock] = self.pct_change[stock].mean()

    def corr(self, stock_a, stock_b):
        if stock_a not in self.stock_list and stock_b not in self.stock_list:
            return None
        return self._corr[stock_a][stock_b]

    def std(self, stock):
        """
        :return: Expected single period volatility
        """
        if stock not in self.stock_list:
            return None
        if stock not in self._std:
            # self.sd[stock] = self.price_data[stock]['adj_close'].std(ddof=0)
            self._std[stock] = self.pct[stock].std(ddof=1)
            # print(self.sd[stock])
        return self._std[stock]

    def total_return(self, stock):
        if stock not in self.stock_list:
            return None
        return self._return[stock]

    def avg_return(self, stock):
        if stock not in self.stock_list:
            return None
        return self._avg_return[stock]

    def checkSD(self):
        """
        Debug use
        """
        for s in self.stock_list:
            print(self.std(s), np.sqrt(self.cov[s][s]))
            if self.std(s) != np.sqrt(self.cov[s][s]):
                return False
        return True

    def get_last_close(self, stock):
        if stock not in self.stock_list:
            return None
        return self.price_data[stock].iloc[-1]['close']

if __name__ == '__main__':
    d = StockData(['MSFT', 'AMZN'], "2018-07-09", "2019-07-14", "d")
    print(d.price_data['MSFT'].tail())
    print(d.price_data['AMZN'].tail())
    print(d.corr('MSFT', 'AMZN'))