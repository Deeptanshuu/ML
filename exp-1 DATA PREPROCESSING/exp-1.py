import numpy as np
import pandas as pd
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

# Displaying preprocessed data
print("Training data after preprocessing:")
print(x_train[:5])  # Display first 5 rows of processed training data
print("Test data after preprocessing:")
print(x_test[:5])  # Display first 5 rows of processed test data