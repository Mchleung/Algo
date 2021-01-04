import json

class Position:
    def __init__(self, asset_id, cost, num_share, date_of_purchase):
        self.asset_id = asset_id
        self.cost = cost
        self.num_share = num_share
        self.date_of_purchase = date_of_purchase

    def to_json(self):
        return json.dumps(self.__dict__)



if __name__ == '__main__':
    p = Position(1299, 74.8, 1000, '2018-09-15')
    print(p.to_json())


