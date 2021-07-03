# load libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# load datset
data_path = 'G:/rauf/STEPBYSTEP/Data/mnist/mnist_digit'
mnist = fetch_openml('mnist_784', data_home=data_path)


# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)


# standarize dataset
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


# define PCA model
# Make an instance of the Model
pca = PCA(.95)


# fit the model
pca.fit(train_img)


# Apply the mapping (transform) to both the training set and the test set.
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


# apply logistic regression to transfromed data
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')


# train the model
logisticRegr.fit(train_img, train_lbl)


# make prediction
# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))


# evaluate the model
logisticRegr.score(test_img, test_lbl)


# in this project we used PCA to improve speed of the model because our data has so many features 
# in case 784 feature sor lowering that dimesion to several PCA will speed up model performance 