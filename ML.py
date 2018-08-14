import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.linspace(0, 4 * np.pi, 100)
f_nonlinear = np.array([[x_i * np.sin(x_i), x_i * np.cos(x_i)] for x_i in X])
A = np.random.normal(scale=1, size=(10, 2))

Y = A.dot(f_nonlinear.T)
variance = 1
print(np.mean(Y, axis=1))
Y = Y - np.mean(Y, axis=1).reshape([-1, 1])


def C(W):
    return W.dot(W.T) + variance * np.eye(N=W.shape[0])


def f(W, *args):
    w = W.reshape(10, 2)
    CW = C(w)
    part1 = Y.shape[1] * np.log(np.linalg.det(CW))
    part2 = np.trace(Y.T.dot(np.linalg.inv(CW)).dot(Y))
    return (part1 + part2)/2


def dfx(W, *args):
    w = W.reshape(10, 2)
    gradients = np.zeros([w.shape[0], w.shape[1]])
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            J = np.zeros([10, 2])
            J[i][j] = 1
            part1 = Y.shape[1] * np.trace(np.linalg.inv(C(w)).dot(w.dot(J.T) + J.dot(w.T)))
            part2 = np.trace(Y.dot(Y.T).dot(-np.linalg.inv(C(w)).dot(w.dot(J.T) + J.dot(w.T)).dot(np.linalg.inv(C(w)))))
            gradients[i][j] = (part1 + part2) / 2
    # return the gradient of the objective at x
    return gradients.reshape([20, 1])[:, 0]


W_init = np.random.randn(20)
w_star = opt.fmin_cg(f, W_init, fprime=dfx)
w_star = w_star.reshape([10, 2])
x = Y.T.dot(np.linalg.pinv(w_star.T))

#plt.scatter(x[:, 0], x[:, 1])
plt.scatter(f_nonlinear[:, 0], f_nonlinear[:, 1], marker='o')
plt.ylabel("x1")
plt.xlabel("x0")
plt.show()
