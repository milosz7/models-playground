import numpy as np
import os
from keras.datasets import cifar10

(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

print(xtrain.shape)

rnd_idx = np.argsort(np.random.random(xtrain.shape[0]))

xtrain = xtrain[rnd_idx]
ytrain = ytrain[rnd_idx]

rnd_idx = np.argsort(np.random.random(xtest.shape[0]))

xtest = xtest[rnd_idx]
ytest = ytest[rnd_idx]

np.save(os.path.join("data", "parsed", "cifar10", "cifar10_training_images.npy"), xtrain)
np.save(os.path.join("data", "parsed", "cifar10", "cifar10_training_labels.npy"), ytrain)
np.save(os.path.join("data", "parsed", "cifar10", "cifar10_test_images.npy"), xtest)
np.save(os.path.join("data", "parsed", "cifar10", "cifar10_test_labels.npy"), ytest)

xtrainv = xtrain.reshape((50000, 32*32*3))
xtestv = xtest.reshape((10000, 32*32*3))

np.save(os.path.join("data", "parsed", "cifar10", "cifar10_training_vectors.npy"), xtrainv)
np.save(os.path.join("data", "parsed", "cifar10", "cifar10_test_vectors.npy"), xtestv)
