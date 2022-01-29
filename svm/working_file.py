import sklearn
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train, y_train)

classes = ['malignant' 'benign']
