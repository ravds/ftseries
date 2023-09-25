import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm

# Function to create dataset with look-back
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# Function to find the best LSTM model for given look_back
def LSTM(log_ret, look_backs=[5, 21, 63, 252], epochs=10, batch_size=32, splits=10):
    best_rmse, best_look_back, best_model = np.inf, None, None

    # Scaling the dataset for the LSTM model
    scaler = StandardScaler()
    dataset = scaler.fit_transform(log_ret.reshape(-1, 1))

    tscv = TimeSeriesSplit(n_splits=splits)

    for look_back in tqdm(look_backs, desc="Evaluating LSTM models", unit="look_back"):
        # Creating dataset with the current look_back
        X, y = create_dataset(dataset, look_back)
        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        rmse_list = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # LSTM model
            model = Sequential()
            model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            # ModelCheckpoint callback
            checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=0)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2, callbacks=[checkpoint])

            # Load best model weights
            model.load_weights('best_model.h5')

            # Making predictions and calculating RMSE
            test_predictions = model.predict(X_test)
            test_predictions = scaler.inverse_transform(test_predictions).flatten()
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions))

            rmse_list.append(rmse)

        avg_rmse = np.mean(rmse_list)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_look_back = look_back
            best_model = model

        print(f"Look-back: {look_back}, Average RMSE: {avg_rmse}")

    return best_rmse, best_look_back, best_model

print(find_best_lstm_model(log_returns))
