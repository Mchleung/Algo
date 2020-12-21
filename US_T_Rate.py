import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
from os.path import isfile, getmtime
import matplotlib.pyplot as plt

def get_interest_rate():
    """
    If the rate csv is updated today, read the csv, else scrape from us gov and save as csv
    :return: Treasury rate dataframe
    """
    # m_date = datetime.datetime.fromtimestamp(getmtime('interest_rate.csv')).strftime('%Y-%m-%d')
    # if isfile('interest_rate.csv') and m_date == datetime.datetime.now().strftime('%Y-%m-%d'):
    #     return pd.read_csv("interest_rate.csv")
    url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldAll'
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find("table", {"class": "t-chart"})
    # print(table)
    table_df = pd.read_html(str(table))[0]
    table_df['Date'] = table_df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d'))
    # print(table_df)

    table_df = table_df.rename(columns={"Date": "date_time"})
    long_rate = get_long_rate()
    fill_dict = long_rate.set_index('date_time')['LT COMPOSITE (>10 yrs)'].to_dict()
    table_df['30 yr'] = table_df['30 yr'].fillna(
        table_df['date_time'].map(fill_dict))  # Fill missing 30 yr from 2002 to 2006
    table_df = table_df.fillna(method='ffill')
    interest_rate_to_csv(table_df)
    table_df = table_df.set_index("date_time")
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

if __name__ == '__main__':
    print(get_interest_rate())

