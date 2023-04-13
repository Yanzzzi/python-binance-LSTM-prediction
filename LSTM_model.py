import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

BASE_URL = 'https://api.binance.com'


def save_as_csv(data: list) -> None:
    pd.DataFrame(data).to_csv('./klines.csv', index=False, header=False)


def get_api_data(api_url: str, **params) -> pd.DataFrame:
    url = BASE_URL + api_url
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


def predict(df: pd.DataFrame) -> np.ndarray:
    training_set = df.iloc[:800, 1:2].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    x_train = []
    y_train = []
    for i in range(60, 800):
        x_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    # add layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    # fitting model
    model.fit(x_train, y_train, epochs=100, batch_size=32)

    dataset_train = df.iloc[:800, 1:2]
    inputs = dataset_train.values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(60, 800):
        x_test.append(inputs[i - 60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # predict
    predicted_btc_price = model.predict(x_test)
    predicted_btc_price = sc.inverse_transform(predicted_btc_price)

    return predicted_btc_price


def draw(predict: np.ndarray, df) -> None:
    plt.plot(df.loc[:, 'Kline open time'], df.iloc[:, 1:2], color='red', label='Real BTCRUB open price')
    plt.plot(df.loc[260:, 'Kline open time'], predict, color='blue', label='Predicted BTCRUB open price')
    plt.title('BTCRUB open price prediction')
    plt.xlabel('Time')
    plt.ylabel('BTCRUB open price')
    plt.legend()
    plt.show()


def main():
    df = get_api_data('/api/v3/klines', symbol='BTCRUB', interval='1h', limit=1000)
    pred = predict(df)
    draw(pred, df)


if __name__ == '__main__':
    main()
