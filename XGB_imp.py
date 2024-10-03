import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,  # Depth of the trees
    'eta': 0.1,  # Learning rate, can experiment with lower values too
    'subsample': 0.8
}


xgb_model = xgb.train(params, dtrain, num_boost_round=100)


predictions_xgb = xgb_model.predict(dtest)

mae_xgb = mean_absolute_error(y_test, predictions_xgb)
r2_xgb = r2_score(y_test, predictions_xgb)

print(f'XGBoost Mean Absolute Error: {mae_xgb}')
print(f'XGBoost R-squared: {r2_xgb}')
