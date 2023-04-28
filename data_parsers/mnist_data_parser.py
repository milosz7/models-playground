import numpy as np
import os
from keras.datasets import mnist

(dtrn, d_idtrn), (dtst, d_idtst) = mnist.load_data()
rnd_idx = np.argsort(np.random.random(dtrn.shape[0]))

dtrn = dtrn[rnd_idx]
d_idtrn = d_idtrn[rnd_idx]

rnd_idx = np.argsort(np.random.random(dtst.shape[0]))

dtst = dtst[rnd_idx]
d_idtst = d_idtst[rnd_idx]

np.save(os.path.join("data", "parsed", "mnist", "mnist_training_data.npy"), dtrn)
np.save(os.path.join("data", "parsed", "mnist", "mnist_training_labels.npy"), d_idtrn)
np.save(os.path.join("data", "parsed", "mnist", "mnist_test_data.npy"), dtst)
np.save(os.path.join("data", "parsed", "mnist", "mnist_test_data_labels.npy"), d_idtst)

dtrnv = dtrn.reshape((60000, 28*28))
dtstv = dtst.reshape((10000, 28*28))

np.save(os.path.join("data", "parsed", "mnist", "mnist_training_data_vectors.npy"), dtrnv)
np.save(os.path.join("data", "parsed", "mnist", "mnist_test_data_vectors.npy"), dtstv)

rnd_idx = np.argsort(np.random.random(28*28))

for i in range(60000):
    dtrnv[i,:] = dtrnv[i, rnd_idx]
for i in range(10000):
    dtstv[i,:] = dtstv[i, rnd_idx]

np.save(os.path.join("data", "parsed", "mnist", "mnist_training_data_vectors_shuffled.npy"), dtrnv)
np.save(os.path.join("data", "parsed", "mnist", "mnist_test_data_vectors_shuffled.npy"), dtrn)

dtrn_shuffled = np.zeros((60000, 28, 28))
dtst_shuffled = np.zeros((10000, 28, 28))


for i in range(60000):
    dtrn_shuffled[i,:,:] = dtrnv[i,:].reshape((28,28))
for i in range(10000):
    dtst_shuffled[i,:,:] = dtstv[i,:].reshape((28,28))

np.save(os.path.join("data", "parsed", "mnist", "mnist_training_data_shuffled_images.npy"), dtrn_shuffled)
np.save(os.path.join("data", "parsed", "mnist", "mnist_test_data_shuffled_images.npy"), dtst_shuffled)
