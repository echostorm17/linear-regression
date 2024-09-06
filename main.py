import pandas as pd
from sklearn.model_selection import train_test_split

# import kaggle

# # Make sure to replace 'mssmartypants/rice-typeclassification' with the correct dataset path
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files('mssmartypants/rice-typeclassification', path='D:/code-test/1/', unzip=True)

# Local path to the downloaded file
file_path = "D:/code-test/1/riceClassification.csv"  # Adjust path as necessary
data = pd.read_csv(file_path)
# print(data.columns)
# Assuming the dataset has features and a binary target variable 'type'
X = data.drop('Class', axis=1)
y = data['Class']
print(X)
print(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'X_train: {X_train}')
print(f'y_train: {y_train}')
print(f'X_test: {X_test}')
print(f'y_test: {y_test}')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize and fit the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict and evaluate
predictions = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

import xgboost as xgb

# Initialize and fit the XGBoost classifier
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_clf.fit(X_train, y_train)

# Predict and evaluate
accuracy = xgb_clf.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_}')

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the distribution for parameters
param_dist = {
    'max_depth': randint(3, 8),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': randint(100, 300),
}

# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=50, cv=3, scoring='accuracy')
random_search.fit(X_train, y_train)

# Best parameters and best score
print(f'Best parameters: {random_search.best_params_}')
print(f'Best cross-validation score: {random_search.best_score_}')

