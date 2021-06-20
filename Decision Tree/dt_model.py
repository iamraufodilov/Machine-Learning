# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder#for train test splitting
from sklearn.model_selection import train_test_split#for decision tree object
from sklearn.tree import DecisionTreeClassifier#for checking testing results
from sklearn.metrics import classification_report, confusion_matrix#for visualizing tree 
from sklearn.tree import plot_tree


# load dataset
df_columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width ', 'species']
df = pd.read_csv('G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Decision Tree/data/iris_training.csv', names=df_columns)
#_>print(df.head())
df.drop(df.index[:1], inplace=True)
#_>print(df.head())
#_>print(df.shape)


# split data
target = df['species']
df1 = df.copy()
df1 = df1.drop('species', axis =1)

# Defining the attributes
X = df1
y = target

# Splitting the data - 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)
#_>print("Training split input- ", X_train.shape)
#_>print("Testing split input- ", X_test.shape)


# Defining the decision tree algorithm
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print('Decision Tree Classifier Created')

# Predicting the values of test data
y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred)) # here we go our prediction on test dataset is 96%