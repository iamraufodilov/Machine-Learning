# load librarries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score


# load dataset
dataset = pd.read_csv('G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Naive Bayes/data/Social_Network_Ads.csv')
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values
#print(X[:5])

# we have character feature so encode it
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
#print(X[:5])

# get the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# feature scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#print(X_train[:5])


# train the model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict test ds
y_pred  =  classifier.predict(X_test)
#print(y_pred)


# evaluate the model
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print(cm)
print(ac)

# oh very good 6 misclassification occurs 
# accuracy is 92.5%