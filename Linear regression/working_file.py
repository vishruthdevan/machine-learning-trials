import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'


X = np.array(data.drop(labels=[predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1)

# best_score = 0
# for _ in range(50):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
#         X, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     print(accuracy)
#     if accuracy > best_score:
#         best_score = accuracy
#         with open("student_linear_model.pickle", "wb") as f:
#             pickle.dump(linear, f)


linear_loaded = pickle.load(open("student_linear_model.pickle", "rb"))

print("coef: ", linear_loaded.coef_)
print("y-intercept: ", linear_loaded.intercept_)
print("accuracy: ", linear_loaded.score(x_test, y_test))

predictions = linear_loaded.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade G3")
pyplot.show()
