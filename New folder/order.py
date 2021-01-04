import datetime
import log
import pandas as pd
from util import data_access
import portfolio


class Order:
    def __init__(self, portfolio_id, order_id, asset, shares, price, order_type, order_datetime, order_status=0):
        self.portfolio_id = portfolio_id
        self.order_id = order_id
        self.asset = asset
        self.shares = shares
        self.price = price
        self.order_type = order_type
        self.order_datetime = order_datetime
        self.order_status = order_status  # 0: Not yet executed, 1: Success, -1: Fail to execute
        self.executed_time = None

    def _update_executed_time(self):
        self.executed_time = datetime.datetime.now()

    def _set_success(self):
        self.order_status = 1
        self._update_executed_time()

    def _set_failed(self):
        self.order_status = -1
        self._update_executed_time()

    def execute(self):
        """
        If order is successfully executed then have to create logs for both Trade account and Cash account
        Else only need log for unsuccessful trade order
        Do nothing if order is not yet expired nor executed
        :return: Order status after execution
        """
        # Checking with server
        # ...
        # TODO: API
        if self.order_status == 1 or self.order_status == -1:
            log.create_log(self.executed_time, 'T', self.order_id)
            if self.order_status == 1:
                log.create_log(self.executed_time, 'C', account_id='something', action='D' if self.shares < 0 else 'W',
                               amount=abs(self.shares * self.price))
        return self.order_status


def get_order(order_id):
    return


def smart_order(portfolio_id, ticker, action='b'):
    order_datetime = datetime.datetime.now()
    p = portfolio.get_portfolio(portfolio_id)
    number, suggested_price = 0, 0
    suggested_period = 'eod'
    if ticker not in p.risky.asset_w.keys():
        # Suggestion based on Monte Carlo VaR?
        pass
    else:
        # Suggestion based on
        pass
    limit_order(portfolio_id, ticker, number, suggested_price, period=suggested_period, action=action).execute()
    return


def basket_trade(portfolio_id, sector, order_datetime=datetime.datetime.now()):
    return


def market_order(portfolio_id, ticker, number, action='b'):
    if action == 's':
        number = -number
    market_o = Order(portfolio_id, '', ticker, number, data_access.get_realtime_quote_from_YahooFinance(ticker),
                     'market', datetime.datetime.now())
    market_o.execute()
    return market_o


def limit_order(portfolio_id, ticker, number, limit_price, period='eod', action='b'):
    if action == 's':
        number = -number
    limit_o = Order(portfolio_id, '', ticker, number, limit_price, 'limit', None)

    return limit_o


def cancel_order(order_id):
    return


if __name__ == '__main__':
    print(data_access.get_realtime_quote_from_YahooFinance('0005.HK'))
