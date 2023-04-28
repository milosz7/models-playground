import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition


def generate_data(start: int, pca: decomposition.PCA, data: np.ndarray):
    original_pca = pca.components_.copy()
    ncomp = pca.components_.shape[0]
    a = pca.transform(data)
    for i in range(start, ncomp):
        pca.components_[i, :] += np.random.normal(
            scale=0.1, size=pca.components_.shape[1])
    augmented_data = pca.inverse_transform(a)
    pca.components_ = original_pca.copy()
    return augmented_data


def main():
    x: np.ndarray = np.load('data/parsed/bc_features_standarized.npy')
    y: np.ndarray = np.load('data/parsed/bc_labels.npy')

    n = int(x.shape[0] * 0.9)

    x_train = x[:n]
    y_train = y[:n]
    x_test = x[n:]
    y_test = y[n:]

    pca = decomposition.PCA(n_components=30)
    pca.fit(x)

    start = 2
    nsets = 10
    nsamples = x_train.shape[0]

    x_new = np.zeros(((nsamples * nsets), x_train.shape[1]))
    y_new = np.zeros(nsamples * nsets)

    for i in range(nsets):
        if (i == 0):
            x_new[0:nsamples, :] = x_train
            y_new[0:nsamples] = y_train
        else:
            x_new[(i * nsamples):(i * nsamples) + nsamples, :] = generate_data(
                start, pca, x_train)
            y_new[(i * nsamples):(i * nsamples) + nsamples] = y_train

    np.save("data/parsed/breast_cancer/bc_data_extended_train.npy", x_new)
    np.save("data/parsed/breast_cancer/bc_data_extended_train_labels.npy", y_new)
    np.save("data/parsed/breast_cancer/bc_data_extended_test.npy", x_test)
    np.save("data/parsed/breast_cancer/bc_data_extended_test_labels.npy", y_test)

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
                label="Augmented data",
                alpha=0.4)
    fig.legend(loc="outside upper right")
    plt.show()


main()