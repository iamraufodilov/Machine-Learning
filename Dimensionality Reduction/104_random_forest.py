# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


# read the data
train=pd.read_csv("G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Dimensionality Reduction/data/train_v9rqX0R.csv")
#print(train.head())


# lets drop unimportant variables
df=train.drop('Item_Outlet_Sales', 1)
df=df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
model = RandomForestRegressor(random_state=1, max_depth=10)
df=pd.get_dummies(df)
model.fit(df,train.Item_Outlet_Sales)


# lets plot feature importsnce graph
features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show() # from the graph it can be seen that mong 11 variables only 4 of them are important features for our model


# lets pick those important features
feature = SelectFromModel(model)
Fit = feature.fit_transform(df, train.Item_Outlet_Sales)