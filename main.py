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

