# ================================
# LSTM Weather Prediction - Model Loader & Predictor (Prediction Only)
# ================================

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Load Dataset 
df = pd.read_csv(r'C:\Users\GECross\Downloads\seattle-weather.csv')
print(df.head())

#Select Target Column 
target = df[["temp_max", "temp_min"]].values

# Scale data 
scaler = MinMaxScaler()
target_scaled = scaler.fit_transform(target)

#  Sequence generator
def make_sequences(data, window=10):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)
WINDOW = 10
X, y = make_sequences(target_scaled, WINDOW)
print("Total samples:", len(X))

# Split into Train/Validation/Test
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val,  y_val  = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Reshape for LSTM
X_train = X_train.reshape((-1, WINDOW, 2))
X_val   = X_val.reshape((-1, WINDOW, 2))
X_test  = X_test.reshape((-1, WINDOW, 2))
# Load trained model
model_path = r"C:\Users\GECross\Downloads\weather_lstm_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Saved model not found. Train it first using the optimized script.")

regressor = load_model(model_path)
regressor.compile(optimizer='adam', loss='mean_squared_error')
print("Model loaded successfully.")

# Make predictions
train_pred_scaled = regressor.predict(X_train)   # shape (n, 2)
val_pred_scaled   = regressor.predict(X_val)
test_pred_scaled  = regressor.predict(X_test)

# Inverse transform to real temperatures
train_pred = scaler.inverse_transform(train_pred_scaled)
val_pred   = scaler.inverse_transform(val_pred_scaled)
test_pred  = scaler.inverse_transform(test_pred_scaled)

y_train_real = scaler.inverse_transform(y_train)
y_val_real   = scaler.inverse_transform(y_val)
y_test_real  = scaler.inverse_transform(y_test)

# Metrics Function 
def print_metrics(y_true, y_pred, name="Data"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{name} RMSE: {rmse:.3f}, R²: {r2:.3f}")

# Print prediction metrics 
print_metrics(y_train_real, train_pred, "Training")
print_metrics(y_val_real,   val_pred,   "Validation")
print_metrics(y_test_real,  test_pred,  "Test")

# Prepare DataFrame for plotting
pred = np.concatenate([train_pred, val_pred, test_pred])

df_pred = pd.DataFrame({
    "actual_max": df["temp_max"][WINDOW:].values,
    "actual_min": df["temp_min"][WINDOW:].values,
    "pred_max": pred[:, 0],
    "pred_min": pred[:, 1],
})

# Add date column
df_pred["date"] = pd.to_datetime(df["date"][WINDOW:].values)

# Slice test range only
df_plot = df_pred.iloc[train_size + val_size:].copy()


plt.figure(figsize=(12,6))
plt.title("Test Results")

plt.plot(df_plot["date"], df_plot["actual_max"], label="actual_max")
plt.plot(df_plot["date"], df_plot["actual_min"], label="actual_min")
plt.plot(df_plot["date"], df_plot["pred_max"],   label="pred_max")
plt.plot(df_plot["date"], df_plot["pred_min"],   label="pred_min")

plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()
# Save predictions
try:
    save_path = r"C:\Users\GECross\Downloads\temperature_predictions.csv"
    df_pred.to_csv(save_path, index=False)
    os.startfile(save_path)
except PermissionError:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_path = rf"C:\Users\GECross\Downloads\temperature_predictions_{timestamp}.csv"
    df_pred.to_csv(alt_path, index=False)
    os.startfile(alt_path)
