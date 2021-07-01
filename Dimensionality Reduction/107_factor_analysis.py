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


# decompose dataset using factor analysis
FA = FactorAnalysis(n_components = 3).fit_transform(df[feat_cols].values)


# visualize result
plt.figure(figsize=(12,8))
plt.title('Factor Analysis Components')
plt.scatter(FA[:,0], FA[:,1])
plt.scatter(FA[:,1], FA[:,2])
plt.scatter(FA[:,2],FA[:,0])
