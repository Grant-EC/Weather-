# ================================
# LSTM Weather Prediction Model
# ================================


# --- Import Libraries ---
import pandas as pd                # For data handling and manipulation
import numpy as np                 # For numerical operations and array manipulations
import matplotlib.pyplot as plt    # For plotting results
import seaborn as sns              # For enhanced plotting aesthetics
import os                          # For file path operations and system commands
from keras.models import Sequential, load_model  # For building or loading LSTM models
from keras.layers import LSTM, Dense, Dropout   # LSTM and dense layers
from sklearn.metrics import mean_squared_error, r2_score  # Metrics for model evaluation
import datetime                    # For timestamping saved files

# --- Load Dataset ---
df = pd.read_csv(r'C:\Users\GECross\Downloads\seattle-weather.csv')  # Load CSV file into DataFrame
df.head()  # Display first few rows to check structure

# --- Basic Data Checks ---
print("Missing values per column:\n", df.isnull().sum())  # Count missing values
print("Duplicate rows:", df.duplicated().sum())           # Count duplicate rows

# --- Select Target Feature ---
training_set = df.iloc[:, 2:3].values  # Extract 3rd column (likely 'temp_max') as target

# --- Convert DataFrame to Training Sequences for LSTM ---
def df_to_XY(data, window_size=10):
    X, y = [], []  # Initialize input sequences and labels
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])  # Previous 'window_size' values as input
        y.append(data[i, 0])                # Current value as label
    return np.array(X), np.array(y)         # Convert lists to numpy arrays

WINDOW = 10  # Number of previous days used for prediction
X, y = df_to_XY(training_set, WINDOW)  # Create sequences
print("Total samples:", len(X))        # Print total number of sequences

# --- Split Data into Train / Validation / Test Sets ---
X_train, y_train = X[:800], y[:800]       # First 800 for training
X_val, y_val = X[800:1000], y[800:1000]   # Next 200 for validation
X_test, y_test = X[1000:], y[1000:]       # Remaining for testing

# --- Reshape Input for LSTM ---
# LSTM expects 3D input: (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# --- Build LSTM Model ---
regressor = Sequential()  # Initialize sequential model

# First LSTM layer + Dropout
regressor.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))  # Drop 20% neurons to reduce overfitting

# Second LSTM layer + Dropout
regressor.add(LSTM(50, return_sequences=True))
regressor.add(Dropout(0.2))

# Third LSTM layer + Dropout
regressor.add(LSTM(50, return_sequences=True))
regressor.add(Dropout(0.2))

# Fourth LSTM layer + Dropout
regressor.add(LSTM(50))
regressor.add(Dropout(0.2))

# Output layer (predict next day's temperature)
regressor.add(Dense(1))

# Compile model with Adam optimizer and mean squared error loss
regressor.compile(optimizer='adam', loss='mean_squared_error')

# --- Load Existing Model or Train New Model ---
model_path = r"C:\Users\GECross\Downloads\weather_lstm_model.h5"

if os.path.exists(model_path):
    # If model file exists, load it
    regressor = load_model(model_path)
    
    # Evaluate loaded model to build metrics (removes warnings)
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    val_loss = regressor.evaluate(X_val, y_val, verbose=0)
    
    history = None  # No training history when loading a pre-trained model
else:
    # Train new model if saved model doesn't exist
    history = regressor.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32
    )
    regressor.save(model_path)  # Save trained model for future use

# --- Training History Visualization ---
if history is not None:
    his = pd.DataFrame(history.history)  # Convert training history to DataFrame
    print(his.head())                     # Show first few rows
    history_loss = his[['loss', 'val_loss']]  # Extract loss metrics
    fig, axes = plt.subplots(1, 1, figsize=(10,5))
    sns.lineplot(data=history_loss, palette="flare")  # Plot training vs validation loss
    plt.title("Training and Validation Loss")
    plt.show()

# --- Make Predictions ---
train_pred = regressor.predict(X_train).flatten()  # Predictions on training set
val_pred = regressor.predict(X_val).flatten()      # Predictions on validation set
test_pred = regressor.predict(X_test).flatten()    # Predictions on test set

# --- Define Function to Print Metrics ---
def print_metrics(y_true, y_pred, name="Data"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
    r2 = r2_score(y_true, y_pred)                        # R-squared score
    print(f"{name} RMSE: {rmse:.3f}, R²: {r2:.3f}")      # Print results

# Print metrics for all datasets
print_metrics(y_train, train_pred, "Training")
print_metrics(y_val, val_pred, "Validation")
print_metrics(y_test, test_pred, "Test")

# --- Combine Actual and Predicted Values into DataFrame ---
pred = np.concatenate([train_pred, val_pred, test_pred])  # Merge all predictions
df_pred = pd.DataFrame(df["temp_max"].copy())            # Copy actual temperature column
df_pred.columns = ["actual"]                             # Rename column
df_pred = df_pred[WINDOW:]                               # Align with prediction indices
df_pred["predicted"] = pred                               # Add predictions

# --- Plot Validation and Test Predictions ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=400)

plt.subplot(2, 1, 1)
plt.title("Validation Results")
sns.lineplot(data=df_pred[800:], palette="flare", alpha=0.8)  # Plot validation predictions

plt.subplot(2, 1, 2)
plt.title("Test Results")
sns.lineplot(data=df_pred[1000:], palette="flare", alpha=0.8)  # Plot test predictions

plt.tight_layout()
plt.show()

# --- Save Predictions to CSV ---
try:
    save_path = r"C:\Users\GECross\Downloads\temperature_predictions.csv"
    df_pred.to_csv(save_path, index=False)  # Save to CSV
    os.startfile(save_path)                  # Open file automatically
except PermissionError:
    # If file is open, save with timestamp to avoid conflict
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_path = rf"C:\Users\GECross\Downloads\temperature_predictions_{timestamp}.csv"
    df_pred.to_csv(alt_path, index=False)
    os.startfile(alt_path)
