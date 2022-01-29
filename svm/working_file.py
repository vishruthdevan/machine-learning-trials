import sklearn
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# print(x_train, y_train)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=3)
clf.fit(x_train, y_train)
y_prediction = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_prediction)
print(accuracy)
print(clf.score(x_train, y_train))


kclf = KNeighborsClassifier(n_neighbors=9)
kclf.fit(x_train, y_train)
print(kclf.score(x_test, y_test))
