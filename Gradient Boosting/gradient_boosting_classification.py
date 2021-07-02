# load libraries
import sklearn
from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier


# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)


# define the model
model = GradientBoostingClassifier()


# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# evaluate the model on the dataset
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores))) # accuracy is 89.9%


# make prediction
model = GradientBoostingClassifier()

# fit the model on the whole dataset
model.fit(X, y)

# make a single prediction
row = [0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]
yhat = model.predict([row])

# summarize prediction
print('Predicted Class: %d' % yhat[0]) # so this bunch of features belong to label 1 sor it predicted correctly