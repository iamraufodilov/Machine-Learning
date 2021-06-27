from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

'''
#creating the bagging classifer 
 ##here we are going to create the bagging classifer and providing the tree as the estimatr##
 
bag_clf = BaggingClassifier( DecisionTreeClassifier(),n_estimators=500,max_samples=100,
bootstrap=True,n_jobs=-1) 

bag_clf.fit(X_train, y_train) 
y_pred = bag_clf.predict(X_test)
'''


# implementing random forest with sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1) 
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"],rnd_clf.feature_importances_):
    print(name, score)