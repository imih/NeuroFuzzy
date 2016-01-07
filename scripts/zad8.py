# coding=utf8

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np

fig = plt.figure()

data = []
x = []
E = []
for file_name in ['z8_t0_n-6', 'z8_t0_n-3', 'z8_t0_n0',
        'z8_t1_n-6', 'z8_t1_n-3', 'z8_t1_n0']:
    data = np.genfromtxt(file_name)
    x = data[:,0]
    E.append(data[:,1])

x = np.linspace(min(x), max(x))

for i in range(len(E)):
    E[i] = E[i][:500]
mu = [0.000001, 0.001, 1]
plt.xlabel('epoha ucenja')
for i in range(len(E) / 3):
    for j in range(3):
        plt.ylim(0, 1000)
        x = np.arange(len(E[3 * i + j]))
        plt.plot(x, E[3 * i + j], label = '$\mu=$' + str(mu[j]))
        plt.title(("Stohasticki " if i == 0 else "Batch " ) + "gradijentni spust")
    plt.legend()
    plt.show()

