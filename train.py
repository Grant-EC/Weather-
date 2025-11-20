# Imports
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load Dataset
df = pd.read_csv(r'C:\Users\GECross\Downloads\seattle-weather.csv')
print(df.head())
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Select Target Feature 
target = df.iloc[:, 2:3].values  # shape (N, 1)

# Scale Data
scaler = MinMaxScaler()
target_scaled = scaler.fit_transform(target)

# Sequence Generator
def make_sequences(data, window=10):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

WINDOW = 10
X, y = make_sequences(target_scaled, WINDOW)

print("Total sequences created:", len(X))

# Train/Val/Test Split
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Reshape for LSTM 
X_train = X_train.reshape((-1, WINDOW, 1))
X_val = X_val.reshape((-1, WINDOW, 1))
X_test = X_test.reshape((-1, WINDOW, 1))

# Build Model
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(WINDOW, 1)))
model.add(Dropout(0.2))

model.add(LSTM(50))  # final LSTM layer
model.add(Dropout(0.2))

model.add(Dense(1))  # output

model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

# Save Model
model_path = r"C:\Users\GECross\Downloads\weather_lstm_model.h5"
model.save(model_path)
print(f"Model saved to: {model_path}")
