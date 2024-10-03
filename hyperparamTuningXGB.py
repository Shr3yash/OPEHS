from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


param_grid = {
    'max_depth': [3, 5, 7],  # Testing various tree depths
    'learning_rate': [0.01, 0.1, 0.2],  # Different learning rates
    'n_estimators': [100, 200]  # Number of trees
}


xgboost_regressor = XGBRegressor()
grid_search = GridSearchCV(estimator=xgboost_regressor, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3)


grid_search.fit(X_train, y_train)


best_xgb_model = grid_search.best_estimator_

print(f'Best XGBoost parameters: {grid_search.best_params_}')
