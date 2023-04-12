import requests
import pandas as pd
import numpy as np
import datetime

# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

BASE_URL = 'https://api.binance.com'


def save_as_csv(data: [list]) -> None:
    pd.DataFrame(data).to_csv('./klines.csv', index=False, header=False)


def get_api_data(api_url: [str], **params) -> pd.DataFrame:
    url = BASE_URL + api_url
    datetime.datetime.today()
    data = requests.get(url, params=params)
    data = data.json()
    save_as_csv(data)

    df = pd.DataFrame(data, dtype=float)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    df.columns = ['Kline open time', 'Open price', 'High price', 'Low price', 'Close price', 'Volume',
                  'Kline Close time', 'Quote asset volume',
                  'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    df['Kline open time'] = pd.to_datetime(df['Kline open time'], unit='ms')
    df['Kline Close time'] = pd.to_datetime(df['Kline Close time'], unit='ms')

    return df


df = get_api_data('/api/v3/klines', symbol='BTCRUB', interval='1h', limit=1000)
print(df)
