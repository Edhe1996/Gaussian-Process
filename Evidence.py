import itertools as itt
import numpy as np
from functools import reduce


def multi(z1, z2):
    return z1 * z2


x1 = [-1, 0, 1]
x2 = x1.copy()
y = [-1, 1]
ittx1 = itt.cycle(x1)
ittx2 = itt.cycle(x2)
itty = itt.cycle(y)
#print(itt1, itt2)
data_X = np.zeros((9, 2))
data_Y = np.zeros((9, 1))
for i in range(9):
    if i < 3:
        data_X[i * 3][0] = next(ittx1)
        data_X[i * 3 + 1][0] = data_X[i * 3][0].copy()
        data_X[i * 3 + 2][0] = data_X[i * 3][0].copy()
    data_X[i][1] = next(ittx2)
    data_Y[i][0] = next(itty)


def model0():
    return 1.0 / 512


def model1(x, y, theta1):
    m = np.zeros(9)
    for j in range(9):
        m[j] = 1. / (1 + np.exp(- y[i][0] * theta1 * x[i][0]))
    return reduce(multi, m)


def model2(x, y, theta2):
    m = np.zeros(9)
    for j in range(9):
        m[j] = 1. / (1 + np.exp(- y[i][0] * (theta2[0] * x[i][0] + theta2[1] * x[i][1])))
    return reduce(multi, m)


def model3(x, y, theta3):
    m = np.zeros(9)
    for j in range(9):
        m[j] = 1. / (1 + np.exp(- y[i][0] * (theta3[0] * x[i][0] + theta3[1] * x[i][1] + theta3[2])))
    return reduce(multi, m)
print(model1(data_X, data_Y, 0.6))
#print(data_X, data_Y)
