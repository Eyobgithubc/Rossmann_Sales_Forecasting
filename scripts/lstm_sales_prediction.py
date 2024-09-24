# lstm_model.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def preprocess_data(df):
   
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    

    df = df.sort_values('Date')
    

    df['CompetitionDistance'].fillna(df['CompetitionDistance'].mean(), inplace=True)
    
    
    df = df[['Date', 'Sales', 'Store', 'DayOfWeek', 'Customers', 'Promo']]
    
    return df

def check_stationarity(df, col_name):
   
    df['Sales_diff'] = df['Sales'].diff().fillna(df['Sales'])
    return df

def create_lagged_features(df, n_lag=30):
 
    features, target = [], []
    for i in range(n_lag, len(df)):
        features.append(df.iloc[i-n_lag:i].values)
        target.append(df.iloc[i]['Sales'])
    return np.array(features), np.array(target)

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(features, target):
   
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    
    n_samples, n_timesteps, n_features = features.shape
    features_reshaped = features.reshape(n_samples * n_timesteps, n_features)
    
    
    features_scaled = scaler.fit_transform(features_reshaped)
    features_scaled = features_scaled.reshape(n_samples, n_timesteps, n_features)
    
   
    target_scaled = scaler.fit_transform(target.reshape(-1, 1))
    
    return features_scaled, target_scaled, scaler




def build_lstm_model(input_shape):
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_loss(history):
   
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

def train_model(model, X_train, y_train):
   
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

  
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop])
    return history

def make_predictions(model, X_test, scaler):
   
    predictions = model.predict(X_test)
    
  
    predictions_rescaled = scaler.inverse_transform(predictions)
    return predictions_rescaled

def calculate_mse(y_true, y_pred):
    
    mse = mean_squared_error(y_true, y_pred)
    return mse
