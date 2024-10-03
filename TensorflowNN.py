import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, r2_score

neural_net_model = Sequential()
neural_net_model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
neural_net_model.add(Dense(32, activation='relu'))  # Second hidden layer
neural_net_model.add(Dense(1))  # Output layer with one unit for regression

neural_net_model.compile(optimizer='adam', loss='mse')

neural_net_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

predictions_nn = neural_net_model.predict(X_test_scaled).flatten()

mae_nn = mean_absolute_error(y_test, predictions_nn)
r2_nn = r2_score(y_test, predictions_nn)

print(f'Neural Network MAE: {mae_nn}')
print(f'Neural Network R²: {r2_nn}')
