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
