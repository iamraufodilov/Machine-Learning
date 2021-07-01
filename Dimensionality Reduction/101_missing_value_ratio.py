# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read the data
train=pd.read_csv("G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Dimensionality Reduction/data/train_v9rqX0R.csv")
#print(train.head())


# now we have to check missing value persentage for each collumn
#print(train.isnull().sum()/len(train)*100) # here you can see only two collumn have missing value in persentage 17% and 28%


# in this section we remove collumn which has more than 20% missing value
# saving missing values in a variable
a = train.isnull().sum()/len(train)*100
# saving column names in a variable
variables = train.columns
variable = [ ]
for i in range(0,12):
    if a[i]<=20:   #setting the threshold as 20%
        variable.append(variables[i])

print(variable) # as you can see now our data has 11 collumn while one collumn removed