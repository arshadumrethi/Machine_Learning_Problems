from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#Load Iris Data and get a description of the data
iris = load_iris()
iris.keys()
print iris['DESCR'][:193] + "\n..."

#Taking a look at the labels
print iris['target_names']
print iris['feature_names']

#Taking a look at the data, the data is stored as a numpy array
print type(iris['data'])
print iris['data'].shape
print iris['data'][:5]
# print iris['target']
# print iris['data']

#Splitting data into Test and Training Sets
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state = 0)

#Calling the KNeighborsClassifier and Fitting it to our data.
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = 1, p = 2, weights = 'uniform')

#Setting a random sample to get a prediction
X_new = np.array([[5, 2.9, 1, 0.2]])

#Getting Prediction and printing results.
prediction = knn.predict(X_new)
print prediction
print iris['target_names'][prediction]

#Printing Accuracy Score. 
print knn.score(X_test, y_test)
