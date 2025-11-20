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
target = df.iloc[:, 2:3].values  # shape (N,1)

# Scale data 
scaler = MinMaxScaler()
target_scaled = scaler.fit_transform(target)

#  Sequence generator
def make_sequences(data, window=10):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
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
X_train = X_train.reshape((-1, WINDOW, 1))
X_val   = X_val.reshape((-1, WINDOW, 1))
X_test  = X_test.reshape((-1, WINDOW, 1))

# Load trained model
model_path = r"C:\Users\GECross\Downloads\weather_lstm_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Saved model not found. Train it first using the optimized script.")

regressor = load_model(model_path)
regressor.compile(optimizer='adam', loss='mean_squared_error')
print("Model loaded successfully.")

# Make predictions
train_pred_scaled = regressor.predict(X_train).flatten()
val_pred_scaled   = regressor.predict(X_val).flatten()
test_pred_scaled  = regressor.predict(X_test).flatten()

# Inverse transform to real temperatures
train_pred = scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
val_pred   = scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
test_pred  = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()

y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_val_real   = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_test_real  = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

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
df_pred = pd.DataFrame(df["temp_max"].copy())
df_pred.columns = ["actual"]
df_pred = df_pred[WINDOW:]  # align
df_pred["predicted"] = pred

# Plot Test Predictions 
fig, axes = plt.subplots(2, 1, figsize=(7, 4), dpi=200)


plt.subplot(1, 1, 1)
plt.title("Test Results")
sns.lineplot(data=df_pred[train_size+val_size:], palette="flare", alpha=1.0)

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

