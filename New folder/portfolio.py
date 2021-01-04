import numpy as np
import pandas as pd
import datetime
from asset import StockData
from optimization import monte_carlo_simulation, scipy_opt
import log
from position import Position
import json

class RiskyPortfolio:
    def __init__(self, logs=[], assets=[], weights=np.array([]), asset_w={}, positions={},
                 start_date=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                 end_date=datetime.datetime.now().strftime("%Y-%m-%d")): # Either logs based or asset+weight based
        self.logs = logs
        self.asset_w = asset_w
        if len(assets) > 0 and asset_w == {}:
            for i in range(len(assets)):
                self.asset_w[assets[i]] = weights[0, i]
        self.start_date = start_date
        self.end_date = end_date
        self.num_iterations = int(1500 * np.sqrt(len(list(self.asset_w.keys()))))
        self.positions = positions
        try:
            self.data = StockData(self.asset_w.keys(), start_date, end_date, period="w")
        except ValueError:
            self.data = pd.DataFrame()

    def positions_to_asset_w(self):
        return

    def build_positions(self):
        if not self.positions:
            pass
        return

    def corr_warning(self, threshold=0.8):
        tmp = self.data._corr
        lc = list(tmp.columns)
        warning_pairs = []
        print(tmp)
        for i in range(len(lc)):
            for j in range(i+1, len(lc)):
                if tmp[lc[i]][lc[j]] > threshold:
                    print("{} and {} are highly correlated.".format(lc[i], lc[j]))
                    warning_pairs.append((lc[i], lc[j]))
        return warning_pairs

    def corr_based_recommendation(self, threshold=0.8, period='d'):
        s_d, e_d = datetime.datetime.strptime(self.start_date, "%Y-%m-%d"), datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        mid_date = datetime.datetime.strftime(s_d + (e_d-s_d)/2, "%Y-%m-%d")
        train_p = RiskyPortfolio(asset_w=self.asset_w, start_date=self.start_date, end_date=mid_date) # Get corr from first half and test
        print("From {} to {}:".format(self.start_date, mid_date))
        warning_pairs = train_p.corr_warning(threshold=threshold)
        warning_list = set(list(sum(warning_pairs, ())))
        r_map = {}
        keep_map = {}
        for s in warning_list:
            r_map[s] = self.data.total_return(s)
            keep_map[s] = 0
        for pair in warning_pairs:
            if r_map[pair[0]] > r_map[pair[1]]:
                keep_map[pair[0]] += 1
            else:
                keep_map[pair[1]] += 1
        print(keep_map)
        drop_list = [asset for asset, value in keep_map.items() if value == 0]
        new_list = list(set(self.asset_w.keys()).difference(set(drop_list)))
        print(new_list)
        original_opt, original_min_var = monte_carlo_simulation(list(self.asset_w.keys()), self.num_iterations, mid_date,
                                                                self.end_date, period=period)
        new_opt, new_min_var = monte_carlo_simulation(new_list, self.num_iterations, mid_date, self.end_date)
        print("New: ")
        new_opt.print_stat()
        print("Old: ")
        original_opt.print_stat()
        print("The new optimal portfolio sharpe ratio: {:.2f}, The old optimal portfolio sharpe ratio: {:.2f}".format(
              new_opt.sharpe_ratio(), original_opt.sharpe_ratio()))
        print("The new optimal portfolio return: {:.2f}%, The old optimal portfolio return: {:.2f}%".format(
            new_opt.total_return() * 100, original_opt.total_return() * 100))
        print("---------------------------------------------------------------------")
        print("The new min-var portfolio std: {:.2f}%, The old min-var portfolio std: {:.2f}%".format(
              new_min_var.annualized_std()*100, original_min_var.annualized_std()*100))
        if new_opt.sharpe_ratio() > original_opt.sharpe_ratio():
            print("Consider dropping:", ', '.join(drop_list))
        return drop_list
    #TODO: THE PROBLEM IS WITH LESS ASSET THE DIVERSIFICATION IS WEAKER EVEN THE ASSETS ARE PAIRWISE HIGHLY CORRELATED
    #TODO: FIX THE PICKING ALGO, NOW IT IS NOW VERY RELIABLE
    #TODO: MAYBE SUGGEST LOW CORR STOCK

    def weighting_based_recommendation(self, period='d'):
        opt, min_var, opt_w, min_var_w = monte_carlo_simulation(stock_list=list(self.asset_w.keys()),
                                                                num_portfolio=self.num_iterations,
                                                                start_date=self.start_date, end_date=self.end_date,
                                                                period=period)
        opt_num_stock = opt.weight_to_num_stock()
        min_var_num_stock = min_var.weight_to_num_stock()
        print("Optimal number of stocks in portfolio:", opt_num_stock)
        print("Optimal number of stocks in min-variance portfolio:", min_var_num_stock)
        for k, v in opt_num_stock.items():
            if v == 0:
                print("Consider dropping {}".format(k))

    def to_position_json(self):
        return json.dumps([x.__dict__ for x in self.positions.values()])


class CashPortfolio:
    def __init__(self, capital=100000, logs=[],
                 start_date=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                 end_date=datetime.datetime.now().strftime("%Y-%m-%d")):
        self.capital = capital
        self.logs = logs



class Portfolio:
    def __init__(self, logs=[], start_date=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                 end_date=datetime.datetime.now().strftime("%Y-%m-%d")):
        self.risky = RiskyPortfolio(start_date=start_date, end_date=end_date)
        self.cash = CashPortfolio()
        self.start_date = start_date
        self.end_date = end_date
        self.logs = logs

    def build_from_logs(self):
        for l in self.logs:
            pass

        return


def get_portfolio(portfolio_id):
    return Portfolio() # Fake


if __name__ == "__main__":
    # s_list = ["WPC", "WELL", "CCI", "PLD", "VER", "AMT", "MPW", "O", "PSA", "EQIX"]
    # s_list = ["V", "MA", "CCI", "PLD", "AAPL", "AMT", "O", "FB", X"OM"]
    s_list = ["GSK", "MSFT", "JPM", "XOM", "LMT", "GE"]
    # p = RiskyPortfolio(assets=s_list, weights=np.random.dirichlet(np.ones(len(s_list)), size=1), start_date="2000-01-01", end_date="2019-07-01")
    # print("Warning pairs:", p.corr_warning(threshold=0.6))
    # dropping = p.corr_based_recommendation(threshold=0.7, period='d')
    # p.weighting_based_recommendation()

    test_p = RiskyPortfolio(positions={1299: Position(1299, 74.8, 1000, '2019-01-01'), 2318: Position(2318, 94.8, 1000, '2018-02-21')})
    print(test_p.to_json())
