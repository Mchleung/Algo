import quandl
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from US_T_Rate import *

quandl.ApiConfig.api_key = "32m3WMYsw9ssVqc6AnB2"

def VIX(vix):
    # vix['Trade Date'] = vix['Trade Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y/%n/%d').strftime('%Y-%m-%d'))
    # print(type(vix))
    print(vix)
    # Interday change of VIX  # Latest Close / Latest Open -1
    print("Interday change of VIX ", vix["Close"].iloc[-1] / vix["Open"].iloc[-1] - 1)

    # Five Days changes # Latest Close(-1) / Close(-5) -1
    print("Five Days change of VIX ", vix["Close"].iloc[-5] / vix["Close"].iloc[-1] - 1)

    # Is VIX > 50?
    print("Is VIX > 50?: ", vix["Close"].iloc[-1] >= 50)

def invert_yeild():
    # return true if have inverted yield in the past week

    # get the yeild rate then save into csv file
    get_interest_rate()

    df = pd.read_csv("interest_rate.csv")

    for i in range(1,7):
        val = df["10 yr"].iloc[-i] - df["3 mo"].iloc[-i]

        if val <= 0:
            print("Inverted Yield Curve in the past week")
            break
            return True

def High_Yield(High_index):
    rolling_min = min(High_index["BAMLH0A0HYM2EY"].iloc[-10:])
    print(rolling_min)
    HY_10rtn = High_index["BAMLH0A0HYM2EY"].iloc[-1] / rolling_min - 1
    print(HY_10rtn)
    print(High_index)

def plot(vix,High_index):
    # fig1 = px.line(vix, x= vix.index, y = "Close")
    # fig1.show()

    fig1 = go.Figure(data=[go.Candlestick(x=vix.index,
                                         open=vix['Open'],
                                         high=vix['High'],
                                         low=vix['Low'],
                                         close=vix['Close'])])
    fig1.update_layout(
        title="One Month VIX Graph ",
        xaxis_title="Date",
        yaxis_title="VIX",
        )

    fig2 = px.line(High_index, x=High_index.index, y='BAMLH0A0HYM2EY')
    fig2.update_layout(
        title="One Month High Yield Index",
        xaxis_title="Date",
        yaxis_title="HY Index",
    )

    with open('Test.html', 'w') as f:
        f.write(fig1.to_html(full_html=True, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=True, include_plotlyjs='cdn'))
        # f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

if __name__ == '__main__':
    # Data Source
    vix = quandl.get("CHRIS/CBOE_VX1", returns="pandas")
    High_index = quandl.get("ML/USTRI", returns="pandas")

    print(vix)

    # VIX(vix)
    #
    # High_Yield(High_index)
    #
    # # Plot the One month Vix Index
    # plot(vix.iloc[-30:], High_index.iloc[-30:])
    #
    # invert_yeild()