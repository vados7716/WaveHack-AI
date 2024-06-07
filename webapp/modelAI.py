import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import requests

keras_load_model("webapp\luckyAI.keras")
async def fetch_coefficients():
    url = 'https://lucky-jet-history.gamedev-atech.cc/public/history/api/history'
    headers = {'session': 'demo'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        coefficients = [item['coefficient'] for item in data[:20]]  # Get the last 20 coefficients
        return coefficients
    else:
        raise Exception(f"Failed to fetch data from the server. Status code: {response.status_code}")

async def parse_coefficients_lucky():
    url = 'https://lucky-jet-history.gamedev-atech.cc/public/history/api/history'
    headers = {'session': 'demo'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        coefficients = [item['coefficient'] for item in data]
        return coefficients
    else:
        raise Exception(f"Failed to fetch data from the server. Status code: {response.status_code}")

async def predict():
    new_coefficients = await parse_coefficients_lucky()
    if len(new_coefficients) < 4:
        raise Exception("Not enough data to make a prediction.")
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    new_coefficients_scaled = scaler.fit_transform(np.array(new_coefficients).reshape(-1, 1))
    
    # Prepare training data
    look_back = 10  # increased look_back window for more stable predictions
    X_train = []
    y_train = []
    for i in range(look_back, len(new_coefficients_scaled)):
        X_train.append(new_coefficients_scaled[i-look_back:i, 0])
        y_train.append(new_coefficients_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Define the model
    model = Sequential()
    model.add(LSTM(512, activation='relu', input_shape=(look_back, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    
    model.fit(X_train_rnn, y_train, epochs=1000, batch_size=32, verbose=0, callbacks=[early_stopping])
    
    X_test = np.array(new_coefficients_scaled[-look_back:])
    X_test_rnn = np.reshape(X_test, (1, X_test.shape[0], 1))
    prediction_scaled = model.predict(X_test_rnn)
    prediction = scaler.inverse_transform(prediction_scaled)
    # Calculate confidence (standard deviation of predictions)
    confidence = np.std(prediction_scaled)
    return prediction[0][0], confidence
