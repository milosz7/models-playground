import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

vector_size = 20

d,d_id = make_classification(n_samples=10000, weights=(0.9,0.1))

idx = np.where(d_id == 0)[0]
c0 = d[idx,:]
c0_id = d_id[idx]

idx = np.where(d_id == 1)[0]
c1 = d[idx,:]
c1_id = d_id[idx]

idx = np.argsort(np.random.random(c0_id.shape))
c0 = c0[idx]
c0_id = c0_id[idx]

idx = np.argsort(np.random.random(c1_id.shape))
c1 = c1[idx]
c1_id = c1_id[idx]

ntrn0 = int(c0.shape[0] * 0.9)
ntrn1 = int(c1.shape[0] * 0.9)

dtrn = np.zeros((ntrn0 + ntrn1, vector_size))
d_idtrn = np.zeros((ntrn0 + ntrn1))

dtrn[:ntrn0] = c0[:ntrn0]
dtrn[ntrn0:] = c1[:ntrn1]
d_idtrn[:ntrn0] = c0_id[:ntrn0]
d_idtrn[ntrn0:] = c1_id[:ntrn1]

n0_rem = int(c0.shape[0] - ntrn0)
n1_rem = int(c1.shape[0] - ntrn1)

dval = np.zeros((int(n0_rem / 2 + n1_rem / 2 ), vector_size))
d_idval = np.zeros(int(n0_rem / 2 + n1_rem / 2 ))

dval[:(n0_rem // 2)] = c0[ntrn0:(ntrn0 + n0_rem // 2)]
d_idval[:(n0_rem // 2)] = c0_id[ntrn0:(ntrn0 + n0_rem // 2)]
dval[(n0_rem // 2):] = c1[ntrn1:(ntrn1 + n1_rem // 2)]
d_idval[(n0_rem // 2):] = c1_id[ntrn1:(ntrn1 + n1_rem // 2)]

dtst = np.concatenate((c0[(ntrn0 + n0_rem // 2):], c1[(ntrn1 + n1_rem // 2):]))
d_idtst = np.concatenate((c0_id[(ntrn0 + n0_rem // 2):], c1_id[(ntrn1 + n1_rem //2):]))

d_zero = d[:,0]
print("statystyki dla zestawu testowego, cecha0:\n")
print("Średnia:", d_zero.mean())
print("Odchylenie:", d_zero.std())
print("Błąd standardowy:", d_zero.std() / np.sqrt(d_zero.shape))
print("Mediana:", np.median(d_zero))
print("Minimalna:", d_zero.min())
print("Maksymalna:", d_zero.max())

plt.boxplot(dtst[:50])
plt.show()