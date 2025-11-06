# ================================
# LSTM Weather Prediction Model
# ================================

# --- Importing Libraries ---
import pandas as pd                     # Data handling and manipulation
import numpy as np                      # Numerical operations and array manipulation
import matplotlib.pyplot as plt         # Visualization of data and results
import os
import sys
import subprocess
# --- Load the Dataset ---
df = pd.read_csv(r'C:\Users\GECross\Downloads\seattle-weather.csv')  
# Reads the Seattle weather dataset into a pandas DataFrame

df.head()                               # Displays the first few rows to inspect the data structure

# --- Basic Data Checks ---
df.isnull().sum()                       # Counts missing (NaN) values in each column
df.duplicated().sum()                   # Checks for any duplicate rows in the dataset

# --- Select the Target Feature ---
training_set = df.iloc[:, 2:3].values   # Extracts the 3rd column (index 2) — likely 'temp_max' — as the target variable
training_set                            # Displays the selected data
len(training_set)                       # Shows number of data points in the dataset

# --- Convert DataFrame to Training Sequences for RNN ---
def df_to_XY(df, window_size=10):
    X_train = []                        # Stores input sequences (features)
    y_train = []                        # Stores corresponding output values (labels)

    # Loop through the dataset to create rolling windows of data
    for i in range(10, len(training_set)):
        X_train.append(training_set[i-10:i, 0])  # Previous 10 days as input sequence
        y_train.append(training_set[i, 0])       # The next (current) day's temperature as label
    
    # Convert to numpy arrays for compatibility with Keras/TensorFlow
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

# --- Generate Input/Output Data ---
WINDOW = 10                             # Number of time steps to look back for each prediction
X, y = df_to_XY(df, WINDOW)             # Create sequences and corresponding labels
print(len(X), len(y))                   # Print total number of samples

# --- Split Data into Train, Validation, and Test Sets ---
X_train = X[:800]                       # First 800 samples for training
y_train = y[:800]
X_val = X[800:1000]                     # Next 200 samples for validation
y_val = y[800:1000]
X_test = X[1000:]                       # Remaining samples for testing
x_test = y[1000:]                       # (Note: this should probably be y_test — lowercase x_test looks like a typo)

# --- Reshape Data for LSTM ---
# LSTMs expect 3D input: (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# --- Build the RNN Model ---
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()                # Initialize a sequential (layer-by-layer) model

# --- Add LSTM Layers ---
# First LSTM layer with dropout regularization to prevent overfitting
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))             # Randomly drops 20% of neurons during training for regularization

# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Fourth and final LSTM layer (no return_sequences since it outputs to Dense layer next)
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# --- Output Layer ---
regressor.add(Dense(units=1))           # Single neuron to predict the next temperature value

# --- Compile the Model ---
regressor.compile(optimizer='adam', loss='mean_squared_error')
# Adam optimizer = efficient adaptive gradient descent
# MSE = suitable loss function for continuous regression

# --- (Optional: Imports for advanced callbacks - commented out) ---
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.optimizers import Adam

# --- Train the Model ---
history = regressor.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),     # Monitors validation performance
    epochs=100,                         # Train for 100 epochs (iterations)
    batch_size=32                       # Train in batches of 32 samples at a time
)
# The model learns to predict the next day's temperature based on the past 10 days.

# --- Convert Training History to DataFrame ---
his = pd.DataFrame(history.history)     # Convert training metrics (loss, val_loss) into a DataFrame
his.head()                              # Show first few rows (useful for inspection)

# --- Visualize Training and Validation Loss ---
import seaborn as sns                   # Visualization library for better-looking plots
his.columns                             # View available metrics ('loss', 'val_loss')
history_loss = his[['loss', 'val_loss']]  # Extract loss columns for plotting

# Plot training and validation loss
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.title("Loss & Val Loss")
sns.lineplot(history_loss, palette="flare")
# This shows how the model’s error decreases over epochs on both training and validation data.

# --- Make Predictions on All Sets ---
train_pred = regressor.predict(X_train).flatten()  # Predictions for training set
val_pred = regressor.predict(X_val).flatten()      # Predictions for validation set
test_pred = regressor.predict(X_test).flatten()    # Predictions for test set

# Concatenate predictions into one array (to align with full dataset)
pred = np.concatenate([train_pred, val_pred, test_pred])

# --- Combine Actual and Predicted Values ---
df_pred = pd.DataFrame(df["temp_max"].copy())      # Copy original temperature data
df_pred.columns = ["actual"]                       # Rename to 'actual'
df_pred = df_pred[WINDOW:]                         # Align indices after sequence generation
df_pred["predicted"] = pred                        # Add predicted column

# --- Plot Results ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=400)

plt.subplot(2, 1, 1)
plt.title("Validation Results")
sns.lineplot(df_pred[800:], alpha=0.8, palette="flare", linestyle=None)
# Shows model’s predictions vs actual values for validation set

plt.subplot(2, 1, 2)
plt.title("Test Results")
sns.lineplot(df_pred[1000:], alpha=0.8, palette="flare", linestyle=None)
# Shows model’s predictions vs actual values for unseen test data

file_path = os.path.abspath("temperature_predictions.csv")  # Get full file path
os.startfile(file_path)