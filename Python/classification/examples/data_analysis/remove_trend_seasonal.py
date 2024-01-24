import sys

sys.path.append("Python/classification/")

from utils import utils
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    train, val, test = utils.load_data()
    combined_datetime = pd.to_datetime(train[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    day_time = int(60*24) # 60 minutes * 24 hours
    trainx = pd.DataFrame()
    for i in train:
        flipped_first_values = np.flip(train[i].head(day_time).values)[int(day_time/2):]
        flipped_last_values = np.flip(train[i].tail(day_time).values)[:int(day_time/2)]
        trainx[i] = np.concatenate([flipped_first_values, train[i].values, flipped_last_values])
    result = seasonal_decompose(trainx['temperature'], model='additive', period= day_time) 
    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    residual = result.resid.dropna()
    data = pd.DataFrame()
    data['temperature'] = train['temperature']
    data['trend'] = trend
    data['seasonal'] = seasonal
    data['residual'] = residual
    data['combined_datetime'] = combined_datetime
    data.set_index('combined_datetime', inplace=True)

    # plt.plot(data['seasonal'].iloc[:1440*3])
    # plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(data['temperature'].iloc[1440*7:1440*15] , label='Original')
    plt.legend(loc='upper left')
    plt.title('Original Time Series')

    plt.subplot(4, 1, 2)
    plt.plot(data['trend'].iloc[1440*7:1440*15] , label='Trend')
    plt.legend(loc='upper left')
    plt.title('Trend Component')

    plt.subplot(4, 1, 3)
    plt.plot(data['seasonal'].iloc[1440*7:1440*15] , label='Seasonal')
    plt.legend(loc='upper left')
    plt.title('Seasonal Component')

    plt.subplot(4, 1, 4)
    plt.plot(data['residual'].iloc[1440*7:1440*15] , label='Residual')
    plt.legend(loc='upper left')
    plt.title('Residual Component')

    plt.tight_layout()
    plt.show()
