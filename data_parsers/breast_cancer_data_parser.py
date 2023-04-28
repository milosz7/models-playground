import numpy as np
import matplotlib.pyplot as plt
import os

data = open("data/raw/wdbc.data", "r")
lines = data.read().splitlines()

labels = ["B", "M"]

data_labels = [labels.index(x.split(",")[1]) for x in lines]
data_labels = np.array(data_labels, dtype="uint8")

data_parsed = [[float(i) for i in x.split(",")[2:]] for x in lines]
data_parsed = np.array(data_parsed)
data_standarized = ((data_parsed - data_parsed.mean(axis=0)) / data_parsed.std(axis=0)) 

rnd_idx = np.argsort(np.random.random(data_labels.shape))

data_labels = data_labels[rnd_idx]
data_parsed = data_parsed[rnd_idx]

np.save(os.path.join("data", "parsed", "bc_features.npy"), data_parsed)
np.save(os.path.join("data", "parsed", "bc_labels.npy"), data_labels)
np.save(os.path.join("data", "parsed", "bc_features_standarized.npy"), data_standarized)


plt.boxplot(data_standarized)
plt.show()
