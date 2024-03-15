# Forex_Price_Prediction

Overview

This Flask web application leverages Long Short-Term Memory (LSTM) neural networks to predict future prices of forex pairs. Utilizing historical data from Yahoo Finance, the app preprocesses data, trains an LSTM model, and offers predictions on future prices. Additionally, it provides trading advice based on the predicted trends.

Features

Historical Data Download: Downloads historical forex pair data over a 10-year period at daily intervals from Yahoo Finance.

Data Preprocessing: Fills missing values, scales data using MinMaxScaler for optimal neural network performance.

LSTM Model Training: Employs LSTM neural networks to understand and predict future forex pair prices.

Prediction and Advice: Predicts future prices for the next 60 days and offers trading advice based on the prediction trend.
