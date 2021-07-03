# load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# load dataset into Pandas DataFrame
data_path = 'G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Principial Component Analysis/data/Iris.csv'
df = pd.read_csv(data_path, names=['sepal length','sepal width','petal length','petal width','target'])
#print(df.head())


# standarize dataset
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
#print(x[:5])

    
# our data has 4 dimesnsions(features) we have to downgrad it to 2 PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
# print(principalDf.head()) # awesome


# lets concatenate target labels to PCA values
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)


# lets visualize 2Dimension PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show() # triple awesome

# in this project we try to use PCA algorithm to visualization problem 
# where some datasets have higher dimesnsion like iris 4 dimesions and it is difficult to visualise 
# so we make two PCA from 4 features and visualize it