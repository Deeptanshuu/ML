import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
data_set = pd.read_csv('Dataset.csv', delimiter=',')

# Ensuring all values are numeric and handling incorrect formatting
data_set = data_set.apply(pd.to_numeric, errors='coerce')

# Extracting independent and dependent variables
x = data_set.iloc[:, 1:].values  # Excluding the 'User' column
y = data_set.iloc[:, 0].values  # Keeping 'User' as the dependent variable

# Handling missing data (Replacing missing data with the mean value)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x)
x = imputer.transform(x)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Implementing Linear Regression
mean_x = np.mean(x_train, axis=0)
mean_y = np.mean(y_train)

# Calculating coefficients
cov_xy = np.sum((x_train - mean_x) * (y_train.reshape(-1, 1) - mean_y), axis=0)
var_x = np.sum((x_train - mean_x) ** 2, axis=0)
beta = cov_xy / var_x
alpha = mean_y - np.dot(mean_x, beta)

# Predicting values
y_pred = alpha + np.dot(x_test, beta)

# Plotting regression against actual data
plt.scatter(x_test[:, 0], y_test, color='red', label='Actual data')
plt.plot(x_test[:, 0], y_pred, color='blue', label='Regression line')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Displaying preprocessed data
print("Training data after preprocessing:")
print(x_train[:5])  # Display first 5 rows of processed training data
print("Test data after preprocessing:")
print(x_test[:5])  # Display first 5 rows of processed test data
