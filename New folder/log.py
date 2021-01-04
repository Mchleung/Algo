
class Logs:
    def __init__(self, datetime, log_type):
        self.datetime = datetime
        self.log_type = log_type


class TradeLogs(Logs):
    def __init__(self, datetime, order_id):
        Logs.__init__(self, datetime, "T")
        self.order_id = order_id


class CashLogs(Logs):
    def __init__(self, datetime, account_id, action, amount):
        Logs.__init__(self, datetime, "C")
        self.account_id = account_id
        self.action = action  # (D)eposit/(W)ithdrawal
        self.amount = amount


def create_log(datetime, log_type, order_id='', account_id='', action='', amount=''):
    if log_type == 'T':
        # some API stuff
        pass
    elif log_type == 'C':
        # some API stuff
        pass
    return
