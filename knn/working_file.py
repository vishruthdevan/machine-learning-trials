import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("knn/car.data")
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
class_ = le.fit_transform(list(data["class"]))


predict = "class"

data = np.array((buying,
                 maint,
                 doors,
                 persons,
                 lug_boot,
                 safety,
                 class_))

data = pd.DataFrame(data.T, columns=[
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class_"]
)

X = data.drop(labels=['class_'], axis=1)
y = data["class_"]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

print(accuracy)
