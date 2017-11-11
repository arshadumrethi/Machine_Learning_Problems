from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#[height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)

_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']

#prediction = clf.predict([[180, 78, 41]])
prediction = clf.predict(_X)
print prediction

acc_tree = accuracy_score(_Y, prediction)
print acc_tree

# #Getting the accuracy for DecisionTree
# pred_tree = clf.predict(X)
# acc_tree = accuracy_score(Y, pred_tree) * 100
# print 'Accuracy for DecisionTree: {}'.format(acc_tree)

# clf_svm = SVC()
# clf_svm.fit(X, Y)
#
# pred_svm = clf_svm.predict([[172, 65, 42]])
# acc_svm = accuracy_score(Y, pred_svm) * 100
#
# print 'SVM prediction is %r' % pred_svm
