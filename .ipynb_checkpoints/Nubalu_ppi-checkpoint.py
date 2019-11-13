import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score


plt.close('all')

# ===========================================
desired_width = 320
pd.set_option('display.width', desired_width)  # Show columns horizontally in console
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)  # Show as many columns as I want in console
pd.set_option('display.max_rows', 1000)  # Show as many rows as I want in console
# ===========================================

# ============================================================================================  Load data

data = pd.read_csv('dataset.csv')
# data.property_price_index.plot.line()

X = data.drop(['property_price_index'], axis=1)  # Features
y = data['property_price_index']  # Target

seed = 123

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# ============================================================================================  Simple Regressor

lr = LinearRegression()

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

lr_error = mse(y_test, y_pred_lr)**(1/2)
lr_accuracy = r2_score(y_test, y_pred_lr)

print('Test set RMSE of linear regressor: {:.2f}'.format(lr_error))
print('Test set score of linear regressor: {:.4f}'.format(lr_accuracy))

# ============================================================================================  Simple RandomForest

rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=seed)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_error = mse(y_test, y_pred_rf)**(1/2)
rf_accuracy = r2_score(y_test, y_pred_rf)

print('Test set RMSE of simple random forest: {:.2f}'.format(rf_error))
print('Test set score of simple random forest: {:.4f}'.format(rf_accuracy))

# importances_rf = pd.Series(rf.feature_importances_, index = X.columns)
# sorted_importances_rf = importances_rf.sort_values()
# sorted_importances_rf.plot(kind='barh', color='lightgreen')
# plt.show()

# ============================================================================================ GridSearch RandomForest

rf_g = RandomForestRegressor(random_state=seed)

params_rf = {'n_estimators': [600, 700, 800], 'max_depth': [3, 4, 5], 'min_samples_leaf': np.arange(0.1, 0.2, 0.02)}

grid_rf = GridSearchCV(estimator=rf_g, param_grid=params_rf, cv=3, scoring='neg_mean_squared_error',
                       verbose=0, n_jobs=-1)

grid_rf.fit(X_train, y_train)

best_hyperparams = grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)

best_model = grid_rf.best_estimator_
y_pred_grid = best_model.predict(X_test)

grid_rf_error = mse(y_test, y_pred_grid)**(1/2)
grid_rf_accuracy = r2_score(y_test, y_pred_grid)

print('Test set RMSE of complex random forest: {:.2f}'.format(grid_rf_error))
print('Test set score of complex random forest: {:.4f}'.format(grid_rf_accuracy))

importances_rf = pd.Series(best_model.feature_importances_, index=X.columns)
sorted_importances_rf = importances_rf.sort_values()
sorted_importances_rf.plot(kind='barh', color='lightgreen')
plt.show()
