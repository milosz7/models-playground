import numpy as np
import os
import matplotlib.pylab as plt

labels = ["Iris-setosa", "Iris-versicolor" ,"Iris-virginica"]

f = open("data/raw/iris.data", "r")
lines = f.read().splitlines()
lines.remove("")

data_indexes = [labels.index(x.split(",")[-1]) for x in lines]
data_indexes = np.array(data_indexes, dtype="uint8")
data_formatted = [[float(j) for j in x.split(",")[:-1]] for x in lines]
data_formatted = np.array(data_formatted)

rnd_idx = np.argsort(np.random.random(data_indexes.shape))

data_indexes = data_indexes[rnd_idx]
data_formatted = data_formatted[rnd_idx]

plt.boxplot(data_formatted)
plt.show()

np.save(os.path.join("data", "parsed", "iris_features.npy"), data_formatted)
np.save(os.path.join("data", "parsed", "iris_labels.npy"), data_indexes)
