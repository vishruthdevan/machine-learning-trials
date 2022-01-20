import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv("student-mat.csv", sep=";")

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences', 'Dalc']]

predict = 'G3'


X = np.array(data.drop(labels=[predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1)


linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

accuracy = linear.score(x_test, y_test)

print(accuracy)


print(linear.coef_)
print(linear.intercept_)


predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])
