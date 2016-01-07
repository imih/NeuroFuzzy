import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('params')
a = data[:,0]
b = data[:,1]
c = data[:,2]
d = data[:,3]

def f(x, a, b):
    return 1.0 /  (1 + np.exp(b  * (x - a)))

for i in range(0, len(a)):
    plt.xlabel('x')
    plt.title('$A_' + str(i+1) + '(x)$')
    x = np.arange(-10.0, 10.0, 0.05)
    plt.plot(x, f(x, a[i], b[i]))
    plt.show()

    plt.xlabel('y')
    plt.title('$B_' + str(i+1) + '(y)$')
    plt.plot(x, f(x, c[i], d[i]))
    plt.show()

