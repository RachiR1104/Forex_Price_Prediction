from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    forex_pair = request.form['forex_pair']
    forex_data = yf.download(forex_pair, period='10y', interval='1d')


    if forex_data.empty:
        return render_template('index.html', error="No data found for the specified Forex pair.")

    forex_data.fillna(method='ffill', inplace=True)
    data_close = forex_data['Close'].values.reshape(-1, 1)

    print(forex_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_close_scaled = scaler.fit_transform(data_close)



    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x_part = data[i:(i + seq_length)]
            y_part = data[i + seq_length]
            xs.append(x_part)
            ys.append(y_part)
        return np.array(xs), np.array(ys)

    sequence_length = 100
    X, y = create_sequences(data_close_scaled, sequence_length)



    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.2)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_denorm = scaler.inverse_transform(y_test)

    rmse = sqrt(mean_squared_error(y_test_denorm, test_predictions))
    print(f"Test RMSE: {rmse}")

    r2 = r2_score(y_test_denorm, test_predictions)
    print(f"Test R² Score: {r2}")

    last_sequence = data_close_scaled[-sequence_length:]
    next_predictions = []
    for _ in range(60):
        last_sequence_scaled = np.array([last_sequence])
        predicted_price = model.predict(last_sequence_scaled)
        next_predictions.append(predicted_price[0, 0])
        last_sequence = np.vstack((last_sequence[1:], predicted_price))

    next_predictions = scaler.inverse_transform(np.array(next_predictions).reshape(-1, 1))

    last_date = forex_data.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60).strftime('%Y-%m-%d').tolist()
    predictions = dict(zip(prediction_dates, next_predictions.flatten().tolist()))
    prediction_dates = list(predictions.keys())
    prediction_values = list(predictions.values())

    return render_template('index.html', prediction_dates=prediction_dates, prediction_values=prediction_values,
                           forex_pair=forex_pair)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x_part = data[i:(i + seq_length)]
        y_part = data[i + seq_length]
        xs.append(x_part)
        ys.append(y_part)
    return np.array(xs), np.array(ys)
@app.route('/predict_and_advise', methods=['GET','POST'])
def predict_and_advise():
    if request.method == 'POST':
        forex_pair = request.form.get('forex_pair')
        if forex_pair:
            forex_data = yf.download(forex_pair, period='10y', interval='1d')
            if forex_data.empty:
                return render_template('index.html', error="No data found for the specified Forex pair.")

            forex_data.fillna(method='ffill', inplace=True)
            data_close = forex_data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_close_scaled = scaler.fit_transform(data_close)

            sequence_length = 100
            X, y = create_sequences(data_close_scaled, sequence_length)

            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.2)
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=75, batch_size=32, validation_data=(X_val, y_val), verbose=1)

            test_predictions = model.predict(X_test)
            test_predictions = scaler.inverse_transform(test_predictions)
            y_test_denorm = scaler.inverse_transform(y_test)

            rmse = sqrt(mean_squared_error(y_test_denorm, test_predictions))
            print(f"Test RMSE: {rmse}")

            last_sequence = data_close_scaled[-sequence_length:]
            next_predictions = []
            for _ in range(60):
                last_sequence_scaled = np.array([last_sequence])
                predicted_price = model.predict(last_sequence_scaled)
                next_predictions.append(predicted_price[0, 0])
                last_sequence = np.vstack((last_sequence[1:], predicted_price))

            next_predictions = scaler.inverse_transform(np.array(next_predictions).reshape(-1, 1))

            last_date = forex_data.index[-1]
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60).strftime('%Y-%m-%d').tolist()
            predictions = dict(zip(prediction_dates, next_predictions.flatten().tolist()))
            prediction_dates = list(predictions.keys())
            prediction_values = list(predictions.values())

            advice = "The trend suggests an increase in value over the 30-day period. It might be a good time to exchange currency." if prediction_values[-1] > prediction_values[0] else "The trend does not suggest a significant increase in value over the 20-day period. You may want to wait for a more favorable rate or consult with a financial expert."

            return render_template('predict_and_advise.html', prediction_dates=prediction_dates, prediction_values=prediction_values, forex_pair=forex_pair, advice=advice)

    return render_template('predict_and_advise.html')



if __name__ == '__main__':
    app.run(debug=True)
