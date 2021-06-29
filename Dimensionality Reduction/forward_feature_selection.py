# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression


# load dataset
train=pd.read_csv("G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Dimensionality Reduction/data/train_v9rqX0R.csv")
#print(train.head())


ffs = f_regression(df,train.Item_Outlet_Sales )

variable = [ ]
for i in range(0,len(df.columns)-1):
    if ffs[0][i] >=10:
       variable.append(df.columns[i])


# in this technique we have to train model firts on only one feature then we ad second the third
# in this scenario we will find feature which will improve model performance more.