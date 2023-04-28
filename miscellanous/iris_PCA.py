import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition

x = np.load("data/parsed/iris_features.npy")[:,:2]
y = np.load("data/parsed/iris_labels.npy")

idxs = np.where(y != 0)
x = x[idxs]

x[:,0] -= x[:,0].mean()
x[:,1] -= x[:,1].mean() 

pca = decomposition.PCA(n_components=2)
pca.fit(x)
variance = pca.explained_variance_ratio_

print("variance ratio:", str(variance))

print(x[:,0])

fig, axs = plt.subplots(2)
axs[0].set_title("PCA with variance")
axs[0].scatter(x[:,0], x[:,1], marker='.', color='b')
x0 = variance[0] * pca.components_[0,0]
y0 = variance[0] * pca.components_[0,1]
axs[0].arrow(0, 0, x0, y0, head_width=0.05, head_length=0.1, facecolor='r', edgecolor='r')
x1 = variance[1] * pca.components_[1,0]
y1 = variance[1] * pca.components_[1,1]
a = pca.transform(x)
axs[0].arrow(0, 0, x1, y1, head_width=0.05, head_length=0.1, facecolor='r', edgecolor='r', label="test")
axs[1].set_title("PCA transformed")
axs[1].scatter(a[:,0], a[:,1], marker=".", color='c')
for ax in axs:
    ax.set(xlabel="$x_0$", ylabel="$x_1$")
    ax.label_outer()
plt.show()
