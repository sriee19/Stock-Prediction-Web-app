import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title('Stock Price Prediction App')

# Define a function to fetch stock data
def get_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# Sidebar - Input parameters
st.sidebar.header('Input Parameters')
symbol = st.sidebar.text_input('Stock Symbol', 'AAPL')
start_date = st.sidebar.text_input('Start Date', '2021-01-01')
end_date = st.sidebar.text_input('End Date', '2022-01-01')

# Fetch stock data
data = get_data(symbol, start_date, end_date)

st.subheader('Raw Data')
st.write(data)

# Predict stock price
st.subheader('Stock Price Prediction')
n_years = st.number_input('Years of Prediction:', 1, 10, 1)
n_days = n_years * 365
df = data[['Adj Close']]
df['Prediction'] = df[['Adj Close']].shift(-n_days)

X = np.array(df.drop(['Prediction'], 1))
X = X[:-n_days]
y = np.array(df['Prediction'])
y = y[:-n_days]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(x_train, y_train)

prediction = model.predict(x_test)
st.subheader('Model Evaluation')
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
st.write(f'Mean Squared Error: {mse}')
st.write(f'R2 Score: {r2}')

# Predict the future stock price
predict_df = data[['Adj Close']].tail(n_days)
prediction = model.predict(predict_df)
st.subheader('Predicted Stock Price')
st.write(predict_df)
st.line_chart(predict_df)

