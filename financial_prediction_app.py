import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title('Personal Financial Prediction App')

# Load personal financial data
# You can upload a CSV file or manually input your financial data.

st.subheader('Financial Data')
financial_data = st.file_uploader('Upload a CSV file', type=['csv'])

if financial_data is not None:
    df = pd.read_csv(financial_data)
    st.write(df)

    # Predict profit or loss percentages
    st.subheader('Profit or Loss Prediction')
    n_years = st.number_input('Years of Prediction:', 1, 10, 1)
    n_days = n_years * 365
    df['Prediction'] = df['Profit'].shift(-n_days)

    X = np.array(df[['InputFeature1', 'InputFeature2']])  # Adjust features accordingly
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

    # Predict profit or loss percentages based on user input
    user_input = st.text_input('Enter Input Feature 1:')
    user_input2 = st.text_input('Enter Input Feature 2:')
    user_prediction = model.predict([[user_input, user_input2]])
    st.subheader('Predicted Profit or Loss Percentage')
    st.write(f'Predicted: {user_prediction[0]}%')
