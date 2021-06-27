# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# reading the data and looking at the first five rows of the data
data=pd.read_csv("G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/K Means/data/Wholesale customers data.csv")
#print(data.head())


# standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)
#print(kmeans.inertia_) # from output you can see inertia is very high which means clustering is too bad


# to find right number of cluster we use elbov method
# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
#plt.show() # from the graph it can be seen that 5-8 can be right number for cluster. lets choose 6 and see result


# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
#print(frame['cluster'].value_counts()) # from result it can be seen that our model grouped our data different six cluster


# in this project clustering method implemented to chose right number of cluster elbov method ben used
# and to assign centroid we use kmeans++ method to better result