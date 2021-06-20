#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# load dataset
dataset = pd.read_csv('G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Logistic Regression/train.csv')
#_>print(dataset.head())


# data  preprocessing

#Checking for missing values
#_>print(dataset.isnull().sum()) # here we go we get missing values in coulmn of age cabin and embarked

#Filling Age column by median
dataset["Age"].fillna(dataset["Age"].median(skipna=True), inplace=True)
#Fillimg Embarked column by the most common port of embarkation
dataset["Embarked"].fillna(dataset['Embarked'].value_counts().idxmax(), inplace=True)
#Dropping the cabin columns
dataset.drop('Cabin', axis=1, inplace=True)


#Dropping unnecessary columns
dataset.drop('PassengerId', axis=1, inplace=True)
dataset.drop('Name', axis=1, inplace=True)
dataset.drop('Ticket',  axis=1, inplace=True)


#Creating variable TravelAlone
dataset['TravelAlone']=np.where((dataset["SibSp"]+dataset["Parch"])>0, 0, 1)
dataset.drop('SibSp', axis=1, inplace=True)
dataset.drop('Parch', axis=1, inplace=True)

# final dataset should look like this 
#_>print(dataset.head())
#_>print(dataset.shape)
#_>print(dataset.info())
#_>print(dataset.describe())


# label encoding for categorical collums eg: sex-> male, female or embarked: S, C ect
#Import label encoder
from sklearn import preprocessing
  
#label_encoder object knows how to understand word labels
label_encoder = preprocessing.LabelEncoder()
  
#Encode labels in column Sex and Embarked
dataset['Sex']= label_encoder.fit_transform(dataset['Sex'])
dataset['Embarked']=label_encoder.fit_transform(dataset['Embarked'])
#_>print(dataset.head()) # here we go our data ready to use


#Setting the value for dependent and independent variables
X = dataset.drop('Survived', 1)
y = dataset.Survived

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


#Fitting the Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


#Prediction of test set
y_pred = lr_model.predict(X_test)
#Predicted values
#_>print(y_pred)


# evaluate the model
#Confusion matrix and classification report
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
#_>print(classification_report(y_test, y_pred))


# here we go our model predicted with 79% accuracy