import numpy as np
import matplotlib.pyplot as plt

r = [[2.767550, 1.395948, 0.256857, 0.654884 ],
[-2.149569, -0.486592, 2.247638, 0.480827],
[2.912786, 0.833915, -2.674430, -0.804692],
[0.784520, -1.246593, -0.442674, 1.267158],
[0.170859, -0.626374, -2.630421, -1.274457]]


def f(x, a, b):
    return 1.0 /  (1 + np.exp(b  * (x - a)))

for i in range(0, len(r)):
    plt.xlabel('x')
    plt.title('A' + str(i+1) + '(x)')
    x = np.arange(-10.0, 10.0, 0.05)
    plt.plot(x, f(x, r[i][0], r[i][1]))
    plt.show()

    plt.xlabel('y')
    plt.title('B' + str(i+1) + '(y)')
    plt.plot(x, f(x, r[i][2], r[i][3]))
    plt.show()

