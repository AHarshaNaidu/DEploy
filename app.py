import datetime
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
import cufflinks as cf
from ta.trend import IchimokuIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Query parameters
start_date = datetime.date(2019, 1, 1)
end_date = datetime.date(2021, 1, 31)

# Ticker symbol selection
tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)  # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)  # Get historical prices

# Ticker information
string_name = tickerData.info.get('longName', 'N/A')

# Ticker data
tickerDf_cleaned = tickerDf.dropna()  # Drop rows with missing values
daily_returns = tickerDf_cleaned['Close'].pct_change()
cumulative_returns = daily_returns.cumsum()

# Bollinger bands
qf = cf.QuantFig(tickerDf, title='First Quant Figure', legend='top', name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)

# Ichimoku Cloud
indicator_ichimoku = IchimokuIndicator(high=tickerDf['High'], low=tickerDf['Low'])
tickerDf['ichimoku_a'] = indicator_ichimoku.ichimoku_a()
tickerDf['ichimoku_b'] = indicator_ichimoku.ichimoku_b()
tickerDf['ichimoku_base_line'] = indicator_ichimoku.ichimoku_base_line()
tickerDf['ichimoku_conversion_line'] = indicator_ichimoku.ichimoku_conversion_line()
fig_ichimoku = go.Figure(data=[go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_a'], name='Ichimoku A'),
                                go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_b'], name='Ichimoku B'),
                                go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_base_line'], name='Base Line'),
                                go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_conversion_line'], name='Conversion Line')],
                            layout=go.Layout(title='Ichimoku Cloud'))

# Stock Price Prediction using LSTM
data = tickerDf['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=64)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Ticker symbol selection
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Fetching data for selected tickers
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

if data.empty:
    print("No data available for selected tickers. Please check your input.")
else:
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(data)
    Sigma = risk_models.sample_cov(data)

    # Perform portfolio optimization
    ef = EfficientFrontier(mu, Sigma)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
    print("Expected Annual Return:", expected_return)
    print("Annual Volatility:", annual_volatility)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Optimized Portfolio Weights:")
    print(pd.Series(cleaned_weights))
