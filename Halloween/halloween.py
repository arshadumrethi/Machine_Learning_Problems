from sklearn import tree
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
import pandas as pd

#Read data in
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train['title']

Y = train['author']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

prediction = clf.predict(test)
print prediction

#Make Dataframe
df = pd.DataFrame(data, columns = ['id', 'text', 'author'])

#Setting Index
df = df.set_index('id')

df['length'] = df.text.str.count(' ')

X = df[df["author"]=="MWS"]["length"].describe()

print X
#Subsetting text as X values
X = df.iloc[:,:1]

print X

for row in X.iterrows():
    print len(row)

#Subsetting author as Y values
Y = df.iloc[:,1:2]

#Calling DecisionTree Algorithm
clf = tree.DecisionTreeClassifier()

#Training or Fitting the model
clf = clf.fit(X, Y)
