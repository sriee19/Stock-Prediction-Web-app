import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title('Cryptocurrency Price Prediction App')

# Load cryptocurrency data
# You can use a data source like 'cryptocompare' or 'ccxt' to fetch cryptocurrency data.

st.subheader('Raw Data')
st.write(crypto_data)

# Predict cryptocurrency price
st.subheader('Cryptocurrency Price Prediction')
n_years = st.number_input('Years of Prediction:', 1, 10, 1)
n_days = n_years * 365
df = crypto_data[['Adj Close']]
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

# Predict the future cryptocurrency price
predict_df = crypto_data[['Adj Close']].tail(n_days)
prediction = model.predict(predict_df)
st.subheader('Predicted Cryptocurrency Price')
st.write(predict_df)
st.line_chart(predict_df)
