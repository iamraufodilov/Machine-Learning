# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import datasets

# read the data
train=pd.read_csv("G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Dimensionality Reduction/data/train_v9rqX0R.csv")
#print(train.head())



lreg = LinearRegression()
rfe = RFE(lreg, 10)
rfe = rfe.fit_transform(df, train.Item_Outlet_Sales)


# in this technique we train model with whole features then we eliminate features one by one and collect model performance one by one
# then some features shows they do not have high impact on model performance, so we have to drop those values