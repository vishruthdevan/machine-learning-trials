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

X = np.array((buying,
             maint,
             doors,
             persons,
             lug_boot,
             safety,
             class_))
print(X.shape)


data = pd.DataFrame(X.T, columns=[
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class_"]
)

print(data.head())
