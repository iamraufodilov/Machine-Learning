# load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# load dataset
dataset = pd.read_csv('G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Random Forest/data/petrol_consumption.csv')
#print(dataset.head())


# prepare dataset
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
#print(X[:5], y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# feature scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#print(X_train[:5])


# train the model
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# here to increase model preformance you can change n_estimators paramaetr which is equal to number of trees in forest
# in this project we use 20 estimators as tree.