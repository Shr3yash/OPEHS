
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Flatten, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Custom RMSE metric for evaluation
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Defining the primary LSTM model architecture
def build_primary_lstm_model():
    model = Sequential()
    model.add(LSTM(128, activation='tanh', input_shape=(X_train_scaled.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))  # Output layer for regression

    model.compile(optimizer=Nadam(learning_rate=0.001), loss='mse', metrics=[root_mean_squared_error])
    return model

# Callbacks to optimize training
early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint_callback = ModelCheckpoint('best_primary_lstm_model.h5', save_best_only=True)

# Building the model and training
primary_lstm_model = build_primary_lstm_model()
history = primary_lstm_model.fit(
    X_train_scaled, y_train,
    epochs=50, 
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop_callback, reduce_lr_callback, checkpoint_callback],
    verbose=1
)

# Generating predictions with the LSTM model
primary_predictions = primary_lstm_model.predict(X_test_scaled).flatten()

# Calculating evaluation metrics for the LSTM model
mae_primary = mean_absolute_error(y_test, primary_predictions)
rmse_primary = np.sqrt(mean_squared_error(y_test, primary_predictions))
r2_primary = r2_score(y_test, primary_predictions)

print(f'Primary LSTM Model - MAE: {mae_primary}, RMSE: {rmse_primary}, R²: {r2_primary}')

# Additional GRU model to compare against LSTM model
def build_gru_model():
    model = Sequential()
    model.add(GRU(128, activation='tanh', input_shape=(X_train_scaled.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(GRU(64, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=[root_mean_squared_error])
    return model

gru_model = build_gru_model()

# Training the GRU model
gru_history = gru_model.fit(
    X_train_scaled, y_train, 
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop_callback, reduce_lr_callback, checkpoint_callback],
    verbose=1
)

# Making predictions with the GRU model
gru_predictions = gru_model.predict(X_test_scaled).flatten()
mae_gru = mean_absolute_error(y_test, gru_predictions)
rmse_gru = np.sqrt(mean_squared_error(y_test, gru_predictions))
r2_gru = r2_score(y_test, gru_predictions)

print(f'GRU Model - MAE: {mae_gru}, RMSE: {rmse_gru}, R²: {r2_gru}')

# Ensemble Predictions: Average of LSTM and GRU models
ensemble_predictions = 0.5 * primary_predictions + 0.5 * gru_predictions
mae_ensemble = mean_absolute_error(y_test, ensemble_predictions)
rmse_ensemble = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
r2_ensemble = r2_score(y_test, ensemble_predictions)

print(f'Ensemble Model - MAE: {mae_ensemble}, RMSE: {rmse_ensemble}, R²: {r2_ensemble}')

# Plotting Loss Curves for LSTM and GRU Models
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Primary LSTM Training Loss')
plt.plot(history.history['val_loss'], label='Primary LSTM Validation Loss')
plt.plot(gru_history.history['loss'], label='GRU Training Loss')
plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss for LSTM and GRU Models')
plt.show()
