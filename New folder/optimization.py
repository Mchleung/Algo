import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
from util.data_access import get_stock_price_from_data_api, get_realtime_quote_from_YahooFinance
from asset import StockData
import time
from scipy import stats
import multiprocessing as mp
from scipy.optimize import minimize

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def lin_reg(x: np.ndarray, y: np.ndarray):
    """
    Linear Regression from statsmodel
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :param x: Series of data
    :param y: Series of data
    :return: Alpha, Beta and Residual Error of OLS regression
    """
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    x = x[:, 1]
    return model.params[0], model.params[1], float(np.std(model.resid))  # Alpha, Beta, Residual Error

class SpecificPortfolio:
    """Tentative Portfolio Structure"""
    def __init__(self, stock_list, weighting_list, data: StockData, period='d', benchmark_data=pd.DataFrame()):
        self.period = period
        self.factor = 252
        if period == 'w':
            self.factor = 52
        if period == 'm':
            self.factor = 12
        self.asset_w = {}  # Key: stock, value: weighting
        for i in range(len(stock_list)):
            if weighting_list.ndim == 2:
                self.asset_w[stock_list[i]] = weighting_list[0, i]
            else:
                self.asset_w[stock_list[i]] = weighting_list[i]
        self.data = data
        self._sd = self._return = self._avg_return = 0
        self.weight_pct = pd.DataFrame()
        self.benchmark_df = self.benchmark_return = benchmark_data  # Using S&P500 as benchmark now
        self.riskless = (1 + 0.02) ** (1 / self.factor) - 1
        self._alpha = self._beta = self._non_sys_risk = 0
        self._w = weighting_list
        self.executed = False

    def exec(self):
        if self._w.ndim == 2:
            self._sd = np.sqrt(np.dot(self._w, np.dot(self.data.cov, self._w.T)))[0][0]
        else:
            self._sd = np.sqrt(np.dot(self._w, np.dot(self.data.cov, self._w.T)))
        for stock in self.asset_w:
            self._return += self.asset_w[stock] * self.data.total_return(stock)
            self.weight_pct[stock] = self.data.pct[stock] * self.asset_w[stock]
        self.weight_pct = self.weight_pct.iloc[1:]
        self.weight_pct['sum'] = self.weight_pct.sum(axis=1)  # Drop first row NaN
        self._return, self._avg_return = self.weight_pct['sum'].sum(), self.weight_pct['sum'].mean()
        self.executed = True
        return self

    def total_return(self) -> float:
        """
        :return: Expected (arithmetic) return in the entire period derived from past data
        """
        if not self.executed:
            self.exec()
        return self._return

    def avg_return(self) -> float:
        """
        :return: Expected (arithmetic) period return derived from past data
        """
        if not self.executed:
            self.exec()
        return self._avg_return

    def annualized_return(self) -> float:
        """
        :return: Expected (arithmetic) annual return derived from past data
        """
        if not self.executed:
            self.exec()
        return self.avg_return() * self.factor

    def std(self) -> float:
        """
        :return: Expected single period volatility (Std. Deviation) from past data
        """
        # Not necessary
        # if self.sd == 0:
        # for s in self.asset_w:
        #     self.sd += (self.asset_w[s] * self.data.get_SD(s)) ** 2
        # for i in self.asset_w:
        #     for j in self.asset_w:
        #         if i != j:
        #             self.sd += self.asset_w[i]*self.asset_w[j]*self.data.get_corr(i,j)
        # self.sd = self.sd ** 0.5
        if not self.executed:
            self.exec()
        return self._sd

    def annualized_std(self) -> float:
        """
        :return: Expected single annualized volatility (Std. Deviation) from past data
        """
        if not self.executed:
            self.exec()
        return self._sd * np.sqrt(self.factor)

    def create_benchmark(self):
        """
        Create benchmark only when required to prevent slowing down simulation.
        """
        if self.benchmark_df.empty:
            self.benchmark_df = get_stock_price_from_data_api("GSPC", self.data.start_date, self.data.end_date,
                                                              "us")  # Using S&P500 as benchmark now
        if self.period == 'm':
            self.benchmark_df = self.benchmark_df.groupby(
                pd.DatetimeIndex(self.benchmark_df.date_time).to_period('M')).nth(0)
        elif self.period == 'w':
            self.benchmark_df = self.benchmark_df.iloc[::5]
        self.benchmark_return = self.benchmark_df['adj_close'].pct_change().iloc[1:]  # Drop first row NaN

    def alpha(self) -> float:
        if not self:
            return 0
        if not self.executed:
            self.exec()
        if self._alpha == 0:
            self.create_benchmark()
            self._alpha, self._beta, self._non_sys_risk = lin_reg(
                np.add(self.benchmark_return.to_numpy(), -self.riskless),
                np.add(self.weight_pct['sum'].to_numpy(), -self.riskless))
        return self._alpha

    def beta(self) -> float:
        if not self:
            return 0
        if not self.executed:
            self.exec()
        if self._beta == 0:
            # self.benchmark_df = get_stock_price_from_data_api("GSPC", self.data.start_date, self.data.end_date, "us") # Using S&P500 as benchmark now
            # if self.period == 'm':
            #     self.benchmark_df = self.benchmark_df.groupby(pd.DatetimeIndex(self.benchmark_df.date_time).to_period('M')).nth(0)
            # elif self.period == 'w':
            #     self.benchmark_df = self.benchmark_df.iloc[::5]
            # self.benchmark_return = self.benchmark_df['adj_close'].pct_change()
            # stock_market_cov = np.cov(self.weight_pct['sum'], self.benchmark_return.iloc[1:])
            # self._beta = stock_market_cov[0, 1] / stock_market_cov[1, 1]
            self.create_benchmark()
            self._alpha, self._beta, self._non_sys_risk = lin_reg(
                np.add(self.benchmark_return.to_numpy(), -self.riskless),
                np.add(self.weight_pct['sum'].to_numpy(), -self.riskless))
        return self._beta

    def tracking_error(self) -> float:
        if not self:
            return 0
        if not self.executed:
            self.exec()
        if self._non_sys_risk == 0:
            self.create_benchmark()
            self._alpha, self._beta, self._non_sys_risk = lin_reg(
                np.add(self.benchmark_return.to_numpy(), -self.riskless),
                np.add(self.weight_pct['sum'].to_numpy(), -self.riskless))
        return self._non_sys_risk

    def sharpe_ratio(self) -> float:
        """
        Measured by excess return divided by standard deviation. \n
        Usage: Evaluate complete investment strategies or diversified portfolios
        """
        if not self:
            return 0
        if not self.executed:
            self.exec()
        return (self.avg_return() - self.riskless) * (self.factor ** 0.5) / self.std()

    def treynor(self) -> float:
        """
        Measured by excess turn divided by beta. \n
        Usage: Rank sub-portfolios/securities of a broader, diversified portfolio
        """
        if not self:
            return 0
        if not self.executed:
            self.exec()
        return (self.avg_return() - self.riskless) * (self.factor ** 0.5) / self.beta()

    def jensen(self) -> float:
        """
        Capture abnormal return, equivalent of alpha in CAPM.
        """
        if not self.executed:
            self.exec()
        return self.alpha()

    def information_ratio(self) -> float:
        """
        Measured by alpha divided by tracking error, i.e. non-systematic risk. \n
        Gauge skill of portfolio manager.
        """
        if not self.executed:
            self.exec()
        return self.alpha() / self.tracking_error()

    def M2_measure(self) -> float:
        """
        Modigliani - Modigliani is a more easily interpreted version of Sharpe Ratio, \n
        It has the same ranking as Sharpe. \n
        :return: M^2 measure, which is risk-adjusted return
        """
        if not self.executed:
            self.exec()
        if self.benchmark_df.empty:
            self.create_benchmark()
        std_m_to_p = np.std(self.benchmark_return.to_numpy()) / self.std()
        r_star = std_m_to_p * self.avg_return() + (1 - std_m_to_p) * self.riskless
        return (r_star - self.benchmark_return.mean()) * self.factor

    def value_at_risk(self, days: int = 1, z: float = 0.95, mode='p') -> float:
        """
        :param days: Period of VaR
        :param z: Percentile
        :param mode: 'p' for parametric, 'h' for historical, 's' for simulation, 'c' for conditional
        :return: Value at risk for portfolio in n days
        """
        factor = np.sqrt(days / 252)
        if not self.executed:
            self.exec()
        if mode == 'p':
            return self.std() * stats.norm.ppf(1 - z) * factor
        elif mode == 'h':
            return np.percentile(self.weight_pct['sum'], 100 * (1 - z)) * factor
        elif mode == 's':
            num_sims = 100000
            return np.percentile(np.random.normal(self.avg_return() * factor, self.std() * factor, num_sims),
                                 100 * (1 - z))
        elif mode == 'c':
            mu_h = 0  # CVaR follows normal linear VaR model for random variable X ~ N(mu_h, sigma_h^2)
            return -(1 - z) ** -1 * stats.norm.pdf(stats.norm.ppf(1 - z)) * self.std() * factor - mu_h
        else:
            raise Exception("Incorrect VaR mode")

    def weight_to_num_stock(self, capital=100000, real_time=False):
        """
        Convert weight to integral number of different stocks (assets) in the portfolio.
        :param capital: Initial capital
        :param real_time: Using real time quote or not
        :return: Dictionary with stock-number pairs
        """
        if not self.executed:
            self.exec()
        asset_num = {}
        for k, v in self.asset_w.items():
            price = get_realtime_quote_from_YahooFinance(k) if real_time else self.data.get_last_close(k)
            asset_num[k] = int(capital * v / price)
        return asset_num

    def asset_risk_contribution(self):
        asset_risk_matrix = np.multiply(self._w.T, self.data.cov * self._w.T) / self.std()
        asset_risk_matrix['Total'] = asset_risk_matrix.dot(self._w.T)
        asset_risk_matrix['Total'] = asset_risk_matrix['Total'].multiply(100 * self._w[0])
        asset_risk_matrix['Risk_Contribution'] = asset_risk_matrix['Total'].divide(self.std())

        return np.array(asset_risk_matrix['Risk_Contribution'])

    def print_stat(self, realtime=False):
        if not self.executed:
            self.exec()
        weighting_df = pd.DataFrame(self.asset_w.items())
        weighting_df.columns = ['ticker', 'weighting']
        num_stock_df = pd.DataFrame(self.weight_to_num_stock().items())
        num_stock_df.columns = ['ticker', 'number of shares']
        print("Max weight: %s" % weighting_df[weighting_df.weighting == weighting_df['weighting'].max()].iloc[0][
            'ticker'])
        print("Min weight: %s" % weighting_df[weighting_df.weighting == weighting_df['weighting'].min()].iloc[0][
            'ticker'])
        # print("Weighting:")
        # print(weighting_df)
        print("Number of shares:")
        init_cap = remaining_cap = 100000
        for s, v in self.weight_to_num_stock(capital=remaining_cap, real_time=realtime).items():
            remaining_cap -= v * (get_realtime_quote_from_YahooFinance(s) if realtime else self.data.get_last_close(s))
        print(num_stock_df)
        print("Annualized return: %f%%" % (self.annualized_return() * 100))
        print("Sharpe ratio: %f" % self.sharpe_ratio())
        print("Standard Deviation: %f%%" % (self.annualized_std() * 100))
        print("Remaining fund: {:.2f} with initial capital {}".format(remaining_cap, init_cap))


def monte_carlo_simulation(stock_list: object, num_portfolio: object, start_date: object, end_date: object, period: object = 'd',
                           allow_short: object = False, data: object = None) -> object:
    """
    Monte Carlo Simulation for portfolio optimization on specific list of stocks. \n
    :param stock_list: List of stock tickers
    :param num_portfolio: Number of iterations
    :param start_date: Date string in format "%Y-%m-%d"
    :param end_date: Date string in format "%Y-%m-%d"
    :param period: 'd' for date, 'w' for week, 'm' for month
    :param allow_short:
    :param data: StockData if provided
    :return: Tuple of optimal portfolio and min-variance portfolio in this run
    """
    if not data:
        data = StockData(stock_list, start_date, end_date, period)
    print(data._corr)

    n = len(stock_list)
    d = {"Volatility": [], "Returns": [], "Sharpe Ratio": []}
    benchmark_data = get_stock_price_from_data_api("GSPC", start_date, end_date, "us")
    factor = 1 if allow_short else 0
    t1 = time.time()
    port_list = [SpecificPortfolio(stock_list,
                                   np.random.dirichlet(np.ones(n), size=1) +
                                   factor * (
                                           np.random.dirichlet(np.ones(n), size=1) - np.random.dirichlet(np.ones(n),
                                                                                                         size=1)),
                                   data, period, benchmark_data)
                 for _ in range(num_portfolio)]
    t2 = time.time()
    print("List comp used {}s".format(t2 - t1))
    p = mp.Pool(processes=mp.cpu_count())
    port_list = list(p.map(SpecificPortfolio.exec, port_list))
    t3 = time.time()
    print("Mapping used {}s".format(t3 - t2))
    for port in port_list:
        d["Volatility"].append(port.annualized_std())
        d["Returns"].append(port.annualized_return())
        d["Sharpe Ratio"].append(port.sharpe_ratio())
    min_var = min(port_list, key=lambda item: item.std())
    opt_port = max(port_list, key=lambda item: item.sharpe_ratio())  # Change this to optimize with other ratio
    print("For loop & opt used {}s".format(time.time() - t3))
    df = pd.DataFrame(d)
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    plt.scatter(x=opt_port.annualized_std(), y=opt_port.annualized_return(), c='red', marker='*', s=80)
    plt.scatter(x=min_var.annualized_std(), y=min_var.annualized_return(), c='blue', marker='D', s=50)
    # plt.yticks(np.arange(0, 1, 0.2))
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.show()
    return opt_port, min_var, opt_port._w, min_var._w


# def neg_sharpe_ratio(stock_list, weighting_list, data: StockData, period='d', benchmark_data=pd.DataFrame()):
#     return -Specific_Portfolio(stock_list, weighting_list, data, period, benchmark_data).sharpe_ratio()
#
#
# def neg_treynor_ratio(stock_list, weighting_list, data: StockData, period='d', benchmark_data=pd.DataFrame()):
#     return -Specific_Portfolio(stock_list, weighting_list, data, period, benchmark_data).treynor()
#
#
# def port_var(stock_list, weighting_list, data: StockData, period='d', benchmark_data=pd.DataFrame()):
#     return Specific_Portfolio(stock_list, weighting_list, data, period, benchmark_data).std()


def scipy_opt(stock_list, start_date, end_date, period='d', benchmark_data=pd.DataFrame(), allow_short=False,
              data=None):
    """
    Scipy minimization for portfolio optimization on specific list of stocks. \n
    :param stock_list: List of stock tickers
    :param start_date: Date string in format "%Y-%m-%d"
    :param end_date: Date string in format "%Y-%m-%d"
    :param period: 'd' for date, 'w' for week, 'm' for month
    :param benchmark_data: The price dataframe from benchmark, e.g. S&P500
    :param allow_short:
    :param data: StockData if provided
    :return: Tuple of optimal portfolio and min-variance portfolio in this run
    """
    if not data:
        data = StockData(stock_list, start_date, end_date, period)

    def opt_sharpe(w):
        return -SpecificPortfolio(stock_list, w, data, period, benchmark_data).sharpe_ratio()

    def opt_treynor(w):
        return -SpecificPortfolio(stock_list, w, data, period, benchmark_data).treynor()

    def min_var(w):
        return SpecificPortfolio(stock_list, w, data, period, benchmark_data).std()

    x0 = np.random.dirichlet(np.ones(len(stock_list)), size=1)
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bnd = [(-1.0 if allow_short else 0.0, 1.0) for _ in range(len(stock_list))]
    opt_sharpe_res = minimize(opt_sharpe, x0, method='SLSQP', options={'disp': True}, constraints=cons, bounds=bnd)
    opt_treynor_res = minimize(opt_treynor, x0, method='SLSQP', options={'disp': True}, constraints=cons, bounds=bnd)
    min_var_res = minimize(min_var, x0, method='SLSQP', options={'disp': True}, constraints=cons, bounds=bnd)
    print(dict(zip(stock_list, list(opt_sharpe_res.x))))
    print(dict(zip(stock_list, list(opt_treynor_res.x))))
    print(dict(zip(stock_list, list(min_var_res.x))))
    return (SpecificPortfolio(stock_list, opt_sharpe_res.x, data, period, benchmark_data),
            SpecificPortfolio(stock_list, min_var_res.x, data, period, benchmark_data),
            opt_sharpe_res.x, min_var_res.x)


def generate_risk_parity_portfolio(stock_list, start_date, end_date, period='d', allow_short=False, data=None):
    """
    Portfolio which all assets contribute same portion of risk to the portfolio. Result obtained by numerical method.\n
    :param stock_list: List of stock tickers
    :param start_date: Date string in format "%Y-%m-%d"
    :param end_date: Date string in format "%Y-%m-%d"
    :param period: 'd' for date, 'w' for week, 'm' for month
    :param allow_short:
    :param data: StockData if provided
    :return: Portfolio and weight of risk parity portfolio in using historical data
    """
    if not data:
        data = StockData(stock_list, start_date, end_date, period)
    length = len(stock_list)
    if not allow_short:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                {'type': 'ineq', 'fun': lambda x: x})
    else:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bnd = [(-1.0 if allow_short else 0.0, 1.0) for _ in range(len(stock_list))]
    x0 = np.random.dirichlet(np.ones(length), size=1)
    x0 = np.ones(length) / length

    def _obj_error(weights):
        target = np.ones(length) / length
        _portfolio = SpecificPortfolio(stock_list, weights, data, period)
        return sum(np.square(_portfolio.asset_risk_contribution() - target))

    equal_risk_res = minimize(_obj_error, x0, method='SLSQP', constraints=cons, bounds=bnd,
                              tol=1e-9, options={'disp': True})
    return SpecificPortfolio(stock_list, equal_risk_res.x, data, period), equal_risk_res.x


def comparison(stock_list, train_st, train_et, test_st, test_et, period, allow_short, benchmark):
    """
    Compare scipy optimization with monte carlo simulation
    :param stock_list: List of stock tickers
    :param train_st: Train start date
    :param train_et: Train end date
    :param test_st: Test start date
    :param test_et: Test end date
    :param period: 'd' for date, 'w' for week, 'm' for month
    :param allow_short:
    :param benchmark: Benchmark ticker
    """
    train_b = get_stock_price_from_data_api(benchmark, train_st, train_et, "us")
    test_b = get_stock_price_from_data_api(benchmark, test_st, test_et, "us")
    s_opt, s_min_variance, s_opt_w, s_min_variance_w = scipy_opt(stock_list, train_st, train_et, period,
                                                                 train_b, allow_short=allow_short)
    print("-----------Scipy Optimal Portfolio-------------")
    s_opt.print_stat()
    print("-----------Scipy Min Variance Portfolio-------------")
    s_min_variance.print_stat()
    start_time = time.time()
    mc_opt, mc_min_variance, mc_opt_w, mc_min_variance_w = monte_carlo_simulation(stock_list,
                                                                                  int(1500 * np.sqrt(len(stock_list))),
                                                                                  train_st, train_et,
                                                                                  period, allow_short=allow_short)
    print("Optimization takes {}s".format(time.time() - start_time))
    print("-----------Monte Carlo Optimal Portfolio-------------")
    mc_opt.print_stat()
    print("-----------Monte Carlo Min Variance Portfolio-------------")
    mc_min_variance.print_stat()

    test_data = StockData(stock_list, test_st, test_et, period)
    print("-----------Scipy Optimal Portfolio Test-------------")
    SpecificPortfolio(stock_list, s_opt_w, test_data, period, test_b).print_stat()
    print("-----------Monte Carlo Optimal Portfolio Test-------------")
    SpecificPortfolio(stock_list, mc_opt_w, test_data, period, test_b).print_stat()
    print("-----------Scipy Min Variance Portfolio Test-------------")
    SpecificPortfolio(stock_list, s_min_variance_w, test_data, period, test_b).print_stat()
    print("-----------Monte Carlo Min Variance Portfolio Test-------------")
    SpecificPortfolio(stock_list, mc_min_variance_w, test_data, period, test_b).print_stat()

    print("-------------Benchmark--------------")
    b_return = test_b['adj_close'].pct_change().sum()
    b_return_avg = test_b['adj_close'].pct_change().mean()
    print("Annualized return: %f%%" % (b_return_avg * 100 * 252))
    b_sd = test_b['adj_close'].pct_change().std(ddof=0)
    print("Sharpe ratio: %f" % ((b_return_avg - 0.02 / 252) / b_sd * (252 ** 0.5)))


def generate_optimized_portfolios(stock_list, train_st, train_et, period, allow_short, benchmark, mode='mc'):
    """
    :param stock_list: List of stock tickers
    :param train_st: Train start date
    :param train_et: Train end date
    :param period: 'd' for date, 'w' for week, 'm' for month
    :param allow_short:
    :param benchmark: Benchmark ticker
    :param mode: mc for Monte Carlo, s for Scipy
    :return: Dictionary of portfolios and weights
    """
    res_dict = {}
    n = len(stock_list)
    data = StockData(stock_list, train_st, train_et, period)
    test_data = StockData(stock_list, '2018-01-01', '2019-07-15', period)
    train_b = get_stock_price_from_data_api(benchmark, train_st, train_et, "us")
    if mode == 'mc':
        opt, min_variance, opt_w, min_variance_w = monte_carlo_simulation(stock_list,
                                                                          int(1500 * np.sqrt(len(stock_list))),
                                                                          train_st, train_et,
                                                                          period, allow_short=allow_short, data=data)
    else:
        opt, min_variance, opt_w, min_variance_w = scipy_opt(stock_list, train_st, train_et, period,
                                                             train_b, allow_short=allow_short, data=data)
    equal_weight = SpecificPortfolio(stock_list, np.ones((1, n)) / n, test_data, period, train_b)
    equal_risk, equal_risk_w = generate_risk_parity_portfolio(stock_list, train_st, train_et, period, allow_short, data)
    inverse_vol_weight = np.array([1 / data.std(x) for x in stock_list])
    inverse_vol_weight = inverse_vol_weight / np.sum(inverse_vol_weight)
    inverse_vol_port = SpecificPortfolio(stock_list, inverse_vol_weight, test_data, period, train_b)

    res_dict['opt_port'] = opt
    res_dict['min_var_port'] = min_variance
    res_dict['equal_port'] = equal_weight
    res_dict['equal_risk_port'] = equal_risk
    res_dict['inverse_vol_port'] = inverse_vol_port
    res_dict['opt_weight'] = opt_w
    res_dict['min_var_weight'] = min_variance_w
    res_dict['equal_weight'] = np.ones((1, n)) / n
    res_dict['equal_risk_weight'] = equal_risk_w
    res_dict['inverse_vol_weight'] = inverse_vol_weight

    return res_dict


if __name__ == "__main__":
    # s_list = ["WPC", "WELL", "CCI", "PLD", "VER", "AMT", "MPW", "O", "PSA", "EQIX"]
    # s_list = ['ARE', 'AMT', 'AIV', 'AVB', 'BXP', 'CCI', 'DLR', 'DRE', 'EQIX', 'EQR', 'ESS', 'EXR', 'FRT', 'HCP', 'HST',
    #           'IRM', 'KIM', 'MAC', 'MAA', 'PLD', 'PSA', 'O', 'REG', 'SBAC', 'SPG', 'SLG', 'UDR', 'VTR', 'VNO', 'WELL', 'WY']
    s_list = ["AAPL", "GE", "DIS", "AMT", "V", "MA"]
    # s_list = ["AAPL","ADBE","AMT","AMZN","AXP","CRM","CSCO","DIS","FB","GOOG",'HD','IBM','JPM','KO','MA','MDT','MRK','MSFT','NDAQ','NFLX','NKE','ORCL','RTN','T','TRIP','TSLA','V','WMT','WYNN']
    benchmark_ticker = "GSPC"
    train_start_date = "2015-01-01"
    train_end_date = "2017-12-31"
    test_start_date = "2018-01-01"
    test_end_date = "2019-06-30"
    period_type = 'w'
    shortable = False
    list_len = len(s_list)

    # p, p_w = generate_risk_parity_portfolio(s_list, train_start_date, test_end_date, period_type, allow_short=False)
    # p.print_stat()
    # print(p.asset_risk_contribution())
    # print(p_w)

    # p_dict = generate_optimized_portfolios(s_list, train_start_date, train_end_date, period_type, shortable,
    #                                        benchmark_ticker, 's')
    # print("===========Opt============")
    # p_dict['opt_port'].print_stat()
    # print(p_dict['opt_weight'])
    # print("===========Min Variance============")
    # p_dict['min_var_port'].print_stat()
    # print(p_dict['min_var_weight'])
    # print("===========Equal Weight============")
    # p_dict['equal_port'].print_stat()
    # print(p_dict['equal_weight'])
    # print("===========Equal Risk============")
    # p_dict['equal_risk_port'].print_stat()
    # print(p_dict['equal_risk_weight'])
    # print("===========Inverse Volatility============")
    # p_dict['inverse_vol_port'].print_stat()
    # print(p_dict['inverse_vol_weight'])

    p = monte_carlo_simulation(s_list, 1000, test_start_date, test_end_date, 'd')
    # p = SpecificPortfolio(s_list, np.ones((1, n))/n, StockData(s_list, train_start_date, test_end_date, 'w'), 'w')
    # print(p.asset_risk_contribution_to_allocation_risk())

    # comparison(s_list, train_start_date, train_end_date, test_start_date, test_end_date, period_type, shortable, benchmark_ticker)

    # test_b = get_stock_price_from_data_api(benchmark_ticker, test_start_date, test_end_date, "us")
    # train_b = get_stock_price_from_data_api(benchmark_ticker, train_start_date, train_end_date, "us")
    # s_opt, s_min_variance, s_opt_w, s_min_variance_w = scipy_opt(s_list, train_start_date, train_tend_date, period_type,
    #                                                              train_b, allow_short=shortable)
    # print("-----------Scipy Optimal Portfolio-------------")
    # s_opt.print_stat()
    # print("-----------Scipy Min Variance Portfolio-------------")
    # s_min_variance.print_stat()
    # start_time = time.time()
    # mc_opt, mc_min_variance, mc_opt_w, mc_min_variance_w = monte_carlo_simulation(s_list,
    #                                                                               int(1500 * np.sqrt(len(s_list))),
    #                                                                               train_start_date, train_end_date,
    #                                                                               period_type, allow_short=shortable)
    # print("Optimization takes {}s".format(time.time() - start_time))
    # print("-----------Monte Carlo Optimal Portfolio-------------")
    # mc_opt.print_stat()
    # print("-----------Monte Carlo Min Variance Portfolio-------------")
    # mc_min_variance.print_stat()
    #
    # test_data = StockData(s_list, test_start_date, test_end_date, period_type)
    # print("-----------Scipy Optimal Portfolio Test-------------")
    # Specific_Portfolio(s_list, s_opt_w, test_data, period_type, test_b).print_stat()
    # print("-----------Monte Carlo Optimal Portfolio Test-------------")
    # Specific_Portfolio(s_list, mc_opt_w, test_data, period_type, test_b).print_stat()
    # print("-----------Scipy Min Variance Portfolio Test-------------")
    # Specific_Portfolio(s_list, s_min_variance_w, test_data, period_type, test_b).print_stat()
    # print("-----------Monte Carlo Min Variance Portfolio Test-------------")
    # Specific_Portfolio(s_list, mc_min_variance_w, test_data, period_type, test_b).print_stat()
    #
    # print("-------------Benchmark--------------")
    # b_return = test_b['adj_close'].pct_change().sum()
    # b_return_avg = test_b['adj_close'].pct_change().mean()
    # print("Annualized return: %f%%" % (b_return_avg * 100 * 252))
    # b_sd = test_b['adj_close'].pct_change().std(ddof=0)
    # print("Sharpe ratio: %f" % ((b_return_avg - 0.02 / 252) / b_sd * (252 ** 0.5)))

    # d_data = StockData(s_list, "2018-01-01", "2019-06-15", "d")
    # w_data = StockData(s_list, "2018-01-01", "2019-06-15", "w")
    # m_data = StockData(s_list, "2018-01-01", "2019-06-15", "m")
    # print("------------Daily-------------")
    # print(d_data.corr)
    # print("------------Weekly-------------")
    # print(w_data.corr)
    # print("------------Monthly-------------")
    # print(m_data.corr)

    # n = len(s_list)
    # pp = Specific_Portfolio(s_list, np.ones((1, n)) / n, StockData(s_list, "2015-01-01", "2019-06-30"))
    # print(pp.value_at_risk(mode='p'))
    # print(pp.value_at_risk(mode='h'))
    # print(pp.value_at_risk(mode='s'))
