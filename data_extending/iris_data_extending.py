import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt


def generate_data(start: int, pca: decomposition.PCA, x: np.ndarray):
    pca_original = pca.components_.copy()
    print(x.shape)
    ncomp = pca.components_.shape[0]
    a = pca.transform(x)

    for i in range(start, ncomp):
        pca.components_[i, :] += np.random.normal(scale=0.1, size=ncomp)

    b = pca.inverse_transform(a)
    pca.components_ = pca_original.copy()
    return b


def main():
    x: np.ndarray = np.load("data/parsed/iris_features.npy")
    y: np.ndarray = np.load("data/parsed/iris_labels.npy")
    pca = decomposition.PCA(n_components=4)
    pca.fit(x)

    n = 120
    aug_start_idx = 2
    nsets = 10

    x_train = x[:n]
    y_train = y[:n]
    x_test = x[n:]
    y_test = y[n:]

    nsample = x_train.shape[0]

    x_new = np.zeros((nsets * nsample, x_train.shape[1]))
    y_new = np.zeros(nsets * nsample, dtype='uint8')

    for i in range(nsets):
        if (i == 0):
            x_new[0:nsample, :] = x_train
            y_new[0:nsample] = y_train
        else:
            x_new[(i * nsample):(i * nsample + nsample), :] = generate_data(
                aug_start_idx, pca, x_train)
            y_new[(i * nsample):(i * nsample + nsample)] = y_train

    idx = np.argsort(np.random.random(nsets * nsample))
    x_new = x_new[idx]
    y_new = y_new[idx]

    np.save("data/parsed/iris/iris_train_features_extended", x_new)
    np.save("data/parsed/iris/iris_train_labels_extended", y_new)
    np.save("data/parsed/iris/iris_test_features_extended", x_test)
    np.save("data/parsed/iris/iris_test_labels_extended", y_test)

    fig = plt.figure(layout="constrained")
    plt.scatter(x_train[:, 0],
                x_train[:, 1],
                marker='o',
                c='b',
                label="Original data")
    plt.scatter(x_new[:, 0],
                x_new[:, 1],
                marker='.',
                c='r',
                label="Augmented data")
    fig.legend(loc="outside upper right")
    plt.show()


main()