import numpy as np
import pandas as pd

def create_features(df, lags=48):
    X = []
    y = []
    for i in range(lags, len(df)-24):
        hist = df['demand_mw'].values[i-lags:i]
        gen_hist = df['generation_mw'].values[i-lags:i]
        price_hist = df['price_inr_mwh'].values[i-lags:i]
        feat = np.concatenate([hist, gen_hist, price_hist, [np.mean(hist), np.std(hist)]])
        X.append(feat)
        y.append(df['demand_mw'].values[i:i+24].mean())
    return np.array(X), np.array(y)

def train_random_forest(X_train, y_train, n_estimators=100):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_prophet(df):
    from prophet import Prophet
    m = Prophet()
    ds = df.reset_index()[['timestamp','demand_mw']].rename(columns={'timestamp':'ds','demand_mw':'y'})
    m.fit(ds)
    future = m.make_future_dataframe(periods=24, freq='H')
    forecast = m.predict(future)
    return m, forecast

def train_lstm(df, lags=48, epochs=10, batch_size=32):
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    X, y = create_features(df, lags=lags)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model