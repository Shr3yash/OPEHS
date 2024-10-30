import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)

# Define model
energy_predictor = Sequential()

# Initial LSTM layer with dropout to reduce overfitting a lil
energy_predictor.add(LSTM(128, activation='tanh', input_shape=(X_train_scaled.shape[1], 1), return_sequences=True))
energy_predictor.add(Dropout(0.3))
energy_predictor.add(BatchNormalization())

# Adding a second LSTM layer with dropout
energy_predictor.add(LSTM(64, activation='tanh', return_sequences=True))
energy_predictor.add(Dropout(0.2))

# Another Dense layer with a lil less units
energy_predictor.add(Dense(32, activation='relu'))
energy_predictor.add(Dropout(0.2))

# Flatten for Dense layers
energy_predictor.add(tf.keras.layers.Flatten())

# Dense layers for final prediction
energy_predictor.add(Dense(16, activation='relu'))
energy_predictor.add(Dropout(0.1))
energy_predictor.add(Dense(1))  # Output layer for regression

# Nadam optimizer for potentially better convergence
energy_predictor.compile(optimizer=Nadam(learning_rate=0.001), loss='mse')

# Early stopping if model's performance plateaus
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Model Training
history = energy_predictor.fit(
    X_train_scaled, y_train, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.2, 
    callbacks=[early_stop, reduce_lr], 
    verbose=1
)

# Model Evaluation
predictions_lstm = energy_predictor.predict(X_test_scaled).flatten()
mae_lstm = mean_absolute_error(y_test, predictions_lstm)
r2_lstm = r2_score(y_test, predictions_lstm)

print(f'LSTM Model MAE: {mae_lstm}')
print(f'LSTM Model R²: {r2_lstm}')

# Additional model with different tuning and layers for comparison
alt_model = Sequential()
alt_model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
alt_model.add(Dropout(0.3))
alt_model.add(BatchNormalization())

alt_model.add(Dense(64, activation='relu'))
alt_model.add(Dropout(0.2))
alt_model.add(BatchNormalization())

alt_model.add(Dense(32, activation='relu'))
alt_model.add(Dense(1))

alt_model.compile(optimizer='adam', loss='mse')

alt_history = alt_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Prediction and metrics for alternative model
predictions_alt = alt_model.predict(X_test_scaled).flatten()
mae_alt = mean_absolute_error(y_test, predictions_alt)
r2_alt = r2_score(y_test, predictions_alt)

print(f'Alternative Model MAE: {mae_alt}')
print(f'Alternative Model R²: {r2_alt}')

# Combine results from models
model_comparison = {
    'LSTM Model': {'MAE': mae_lstm, 'R2': r2_lstm},
    'Alternative Model': {'MAE': mae_alt, 'R2': r2_alt}
}

print("\nComparison of model performances:")
for model, metrics in model_comparison.items():
    print(f"{model} - MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")

# Analysis of training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='LSTM Training Loss')
plt.plot(history.history['val_loss'], label='LSTM Validation Loss')
plt.plot(alt_history.history['loss'], label='Alt Training Loss')
plt.plot(alt_history.history['val_loss'], label='Alt Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Comparison')
plt.show()

# Ensemble Prediction - combining predictions from both models
ensemble_predictions = 0.5 * predictions_lstm + 0.5 * predictions_alt
mae_ensemble = mean_absolute_error(y_test, ensemble_predictions)
r2_ensemble = r2_score(y_test, ensemble_predictions)

print(f'Ensemble Model MAE: {mae_ensemble}')
print(f'Ensemble Model R²: {r2_ensemble}')
