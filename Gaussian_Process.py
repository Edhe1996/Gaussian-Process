import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
#import GPy
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def rbf(x1, x2, sigma, lengthscale):
    return sigma * np.exp(-0.5 * cdist(x1, x2)**2/lengthscale)


def posterior_compute(x_new, x_train, y, para):
    a = rbf(x_new, x_train, para[0], para[1])
    b = rbf(x_new, x_new, para[0], para[1])
    c = rbf(x_train, x_train, para[0], para[1])
    # d = rbf(x_train, x_new, para[0], para[1])
    d = a.T
    Sinv = np.linalg.inv(c)
    mu = np.dot(a, Sinv).dot(y)
    m = np.dot(a, Sinv).dot(d)
    sigma = b - m
    return mu.squeeze(), sigma.squeeze()


# Parameters
theta = [1., 1.]
# k = GPy.kern.RBF(d, variance=var, lengthscale=theta)
# _, axes = plt.subplots(3, 1, figsize=(10, 10))
# k = GPy.kern.RBF(input_dim=1, variance=var, lengthscale=0.2)

ori_X = np.linspace(- 2 * np.pi, 2 * np.pi, 500)
X = ori_X.reshape(-1, 1)
prior_Sigma = rbf(X, X, theta[0], theta[1])
# print(np.shape(prior_k))
# print(np.shape(X))
prior_Mu = np.zeros(500)
# C = k.K(X, X)

prior = np.random.multivariate_normal(prior_Mu, prior_Sigma, 20)

fig = plt.figure()
#for i in range(5):
    #plt.plot(X[:], prior[i, :])
# plt.matshow(C)

data_X = np.linspace(- 1 * np.pi, 1 * np.pi, 7)
data_X = data_X.reshape(-1, 1)
data_Y = np.sin(data_X) + np.random.normal(loc=0.0, scale=np.sqrt(0.5), size=(7, 1))


far_X = np.linspace(2 * np.pi, 3 * np.pi, 500)
far_X = far_X.reshape(-1, 1)

#data_k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

# posterior = GPy.models.GPRegression(data_X, data_Y, data_k)
# posterior.plot()
para = [1.0, 0.5]
posterior_Mu, posterior_Sigma = posterior_compute(X, data_X, data_Y, para)
posterior = np.random.multivariate_normal(posterior_Mu, posterior_Sigma, 30)
for i in range(10):
    plt.plot(X[:], posterior[i, :])
plt.plot(data_X, data_Y, 'ko', mew=1.5)
Y1, Y2 = np.meshgrid(X, X)
#print(posterior_Mu.shape)
fig2 = plt.figure()
# plt.matshow(posterior_Mu)
plt.matshow(posterior_Sigma)
fig3 = plt.figure()
ax = fig.gca(projection='3d')
prediction = [posterior_compute(np.array([i]), data_X, data_Y, para) for i in X]
Mu, diag_Sigma = np.transpose(prediction)
#print(np.shape(Mu), np.shape(diag_Sigma))
#plt.fill_between(ori_X, Mu + diag_Sigma, Mu - diag_Sigma, color='#aaaaaa')
#plt.plot(X, posterior_Mu, 'r')
#plt.errorbar(X, Mu, yerr=diag_Sigma, ecolor='grey')
#plt.plot(data_X, data_Y, 'ko', mew=1.5)
#plt.plot(X, posterior_Mu, 'r')
surf = ax.plot_surface(Y1, Y2, posterior_Sigma, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()
