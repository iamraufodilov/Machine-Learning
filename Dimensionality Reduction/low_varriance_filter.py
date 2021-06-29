# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read the data
train=pd.read_csv("G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Dimensionality Reduction/data/train_v9rqX0R.csv")
#print(train.head())


# first we have to fill missing value data
train['Item_Weight'].fillna(train['Item_Weight'].median(), inplace=True)
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)

a = train.isnull().sum()/len(train)*100
#print(a) # so now you can see we do not have missing value

# lets calculate variance for all collumns
#print(train.var()) # from result it can be seen that 'Item_Visibility' has very low varriance so we have to remove that collumn

numeric = train[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']]
var = numeric.var()
numeric = numeric.columns
variable = [ ]
for i in range(0,len(var)):
    if var[i]>=10:   #setting the threshold as 10%
       variable.append(numeric[i])

print(variable) # from result you can see item visibility is removed from collumn list
