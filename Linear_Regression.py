import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from matplotlib.mlab import bivariate_normal
from numpy.linalg import inv


w0, w1 = -1.3, 0.5
vari = 0.3
alpha = 2.0
s = np.sqrt(vari)
beta = 1 / vari
xx = np.linspace(-1, 1, 201)
tar = w0 * xx + w1 + norm.rvs(loc = 0, scale = s, size =(201,))
w0, w1 = np.mgrid[-3:3:.02, -3:3:.02]

fi = np.vstack((np.ones(201), xx)).T
num = [0, 1, 50, 150]
S = inv(alpha * np.eye(2))
Mu = [0, 0]
_, axes = plt.subplots(4, 2, figsize=(8, 12))
for n in range(4):
    # ran = np.random.randint(0, 201, 3)
    # i = np.array([0, ran[0], ran[1], ran[2]])
    # n_fi = np.array([fi[x] for x in i])
    # n_tar = np.array([tar[x] for x in i])
    # print(n_fi)
    # print(n_tar)
    i = num[n]
    likelihood = 1 / np.sqrt(2 * np.pi * vari) * np.exp(-(tar[i] - (w0 * xx[i] + w1)) ** 2 / (2 * vari))

    S = inv(alpha * np.eye(2) + beta * np.dot(fi[:i].T, fi[:i]))
    Mu = beta * np.dot(np.dot(S, fi[:i].T), tar[:i])

    prpo = bivariate_normal(w0, w1, mux=Mu[0], muy=Mu[1], sigmax=np.sqrt(S[0][0]), sigmay=np.sqrt(S[1][1]), sigmaxy=S[0][1])
    ww = multivariate_normal.rvs(mean=Mu, cov=S, size=6)
    ww0, ww1 = ww[:, 0], ww[:, 1]

    axes[n][0].contourf(w0, w1, prpo)
    axes[n][0].set_xticks([-3, 0, 3])
    axes[n][0].set_ylim(-3, 3)
    axes[n][0].set_yticks([-3, 0, 3])
    axes[n][0].text(2.1, -3.8, '$w_0$', fontsize='xx-large')
    axes[n][0].text(-3.9, 2.3, '$w_1$', fontsize='xx-large')
    # axes[n][1].plot(xx[:i], tar[:i], 'o')
    for ind in range(6):
        axes[n][1].plot(w0, ww0[ind] + ww1[ind] * w0, 'r')

#prior = np.random.multivariate_normal(Mu, S, 201)
#prior = bivariate_normal(w0, w1, mux=Mu[0], muy=Mu[1], sigmax=np.sqrt(S[0][0]), sigmay=np.sqrt(S[1][1]), sigmaxy=S[0][1])
#n = random.randint(0, 201)
#likelihood = 1 / np.sqrt(2 * np.pi * vari) * np.exp(-(tar[n] - (w0 * xx[n] + w1)) ** 2 / (2 * vari))
#S2 = inv(alpha * np.eye(2) + beta * np.dot(fi[n].T, fi[n]))
#Mu2 = beta * np.dot(np.dot(S, fi[n].T), tar[n])
#posterior = bivariate_normal(w0, w1, mux=Mu2[0], muy=Mu2[1], sigmax=np.sqrt(S2[0][0]), sigmay=np.sqrt(S2[1][1]), sigmaxy=S2[0][1])
#axes[0].contourf(w0, w1, prior)
#axes[0].set_xticks([-3, 0, 3])
#axes[0].set_ylim(-3, 3)
#axes[0].set_yticks([-3, 0, 3])
#axes[0].text(2.5, -3.5, '$w_0$', fontsize='xx-large')
#axes[0].text(-3.5, 2.5, '$w_1$', fontsize='xx-large')

#axes[1].contourf(w0, w1, posterior)
#axes[1].set_xticks([-3, 0, 3])
#axes[1].set_ylim(-3, 3)
#axes[1].set_yticks([-3, 0, 3])
#axes[1].text(2.5, -3.5, '$w_0$', fontsize='xx-large')
#axes[1].text(-3.5, 2.5, '$w_1$', fontsize='xx-large')

plt.show()
