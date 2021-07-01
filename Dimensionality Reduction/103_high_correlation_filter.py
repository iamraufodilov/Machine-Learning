# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read the data
train=pd.read_csv("G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Dimensionality Reduction/data/train_v9rqX0R.csv")
#print(train.head())


#first we have to remove dependant variable
df=train.drop('Item_Outlet_Sales', 1)
print(df.corr()) # from result it can be seen that we do not have high correlated data if variables high correlated it shows the number grater than treshold 0.5 or 0.6
