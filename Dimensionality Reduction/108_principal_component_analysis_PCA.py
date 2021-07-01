# load libraries
import pandas as pd
import numpy as np
from glob import glob
import cv2
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt


# load data
images = [cv2.imread(file) for file in glob('G:/rauf/STEPBYSTEP/Data/mnist/train/*.png')]


# convert images to numpy arry so we can make matematicl operations
images = np.array(images)
print(images.shape) # image have 3 dimesnion


# we have to convert image to one dimension
image = []
for i in range(0,60000):
    img = images[i].flatten()
    image.append(img)
image = np.array(image)


# lets create dataframe to contain every pixel value of image and their corresponding value
train = pd.read_csv("G:/rauf/STEPBYSTEP/Data/mnist/train.csv")
feat_cols = [ 'pixel'+str(i) for i in range(image.shape[1]) ]
df = pd.DataFrame(image,columns=feat_cols)
df['label'] = train['label']
 

# randomly plot some images from our dataset
rndperm = np.random.permutation(df.shape[0])
plt.gray()
fig = plt.figure(figsize=(20,10))
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1)
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28*3)).astype(float))


# implement pca using python
from sklearn.decomposition import PCA
pca = PCA(n_components=4) # here n components are number of principial components
pca_result = pca.fit_transform(df[feat_cols].values)