# load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# load dataset
dataset = pd.read_csv('G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Random Forest/data/bill_authentication.csv')
print(dataset.head())


#prepare dataset
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# train the model
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# evaluate the model
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# accuracy is 98.2% which is good, so no need to change parameters.