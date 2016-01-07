# coding=utf8

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np

fig = plt.figure()

data = np.genfromtxt('z7_batch')
x = data[:,0]
E_b = data[:,1]

data = np.genfromtxt('z7_stoch')
E_s = data[:,1]

x = np.linspace(min(x), max(x))
x = np.arange(len(E_b))
plt.xlabel('epoha ucenja')
plt.plot(x, E_b, label = 'batch gradijentni spust')
plt.plot(x, E_s, label = 'stohasticki gradijentni spust')
plt.legend()
plt.show()

