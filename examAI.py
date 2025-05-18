import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (x-2)**2

def fprim(x):
    return 2*x - 4


def gradient(x0, etha):
    xn = x0
    for i in range(10):
        xn = xn - etha*fprim(xn)
        plt.scatter(xn, f(xn), label= str(i))

X = np.arange(0,4, 0.01)
plt.plot(X, [f(x) for x in X], color = "black")
gradient(3, 1.02)
plt.legend()
plt.show()






