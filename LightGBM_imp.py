import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score


train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

lightgbm_params = {
    'objective': 'regression',  # Regression task
    'metric': 'mae',  # Mean absolute error for evaluation
    'learning_rate': 0.1,  # Step size of gradient descent
    'num_leaves': 31  # Controls model complexity
}

lgbm_model = lgb.train(lightgbm_params, train_data, num_boost_round=100)


lgbm_predictions = lgbm_model.predict(X_test)


mae_lgbm = mean_absolute_error(y_test, lgbm_predictions)
r2_lgbm = r2_score(y_test, lgbm_predictions)

print(f'LightGBM Mean Absolute Error: {mae_lgbm}')
print(f'LightGBM R-squared: {r2_lgbm}')
