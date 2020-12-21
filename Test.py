from Back_Test import *
# from Momentum import *
import math
from SPX_Historical_Constituents import *
import datetime
import re

def main():
    # df, SPY_con = process_constituents()
    # # SPY_con["Date"].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d'))
    # SPY_con.index = SPY_con["Date"]
    # # print(SPY_con)
    # # print(SPY_con.truncate(after='20201029').tail(1))

    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/Mining-BTC-180.csv")

    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}]]
               # [{"type": "scatter"}],
               # [{"type": "scatter"}]]
    )

    # fig.add_trace(
    #     go.Scatter(
    #         x=df["Date"],
    #         y=df["Mining-revenue-USD"],
    #         mode="lines",
    #         name="mining revenue"
    #     ),
    #     row=3, col=1
    # )
    #
    # fig.add_trace(
    #     go.Scatter(
    #         x=df["Date"],
    #         y=df["Hash-rate"],
    #         mode="lines",
    #         name="hash-rate-TH/s"
    #     ),
    #     row=2, col=1
    # )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Date", "Number<br>Transactions"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[["Mar","Feb"],[1,2]],
                align="left")
        ),
        row=1, col=1
    )
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Bitcoin mining stats for 180 days",
    )

    fig.show()
    with open("C:/Users/user/OneDrive - HKUST Connect/Trading/Quant Program/Backtest/"
              + 'BackTest ' + '.html', 'w') as f:
        f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))


if __name__ == '__main__':
    main()
    # test()