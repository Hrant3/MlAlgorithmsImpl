

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class t_SNE:
    def __init__(self, X):
        self.X = X
    def compute_pairwise_affinities(self,X,perplexity):
        # computing p(j | i) and p(i | j) for each pair of points using variances for each datapoint

        p_i_given_j = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            diff = X[i] - X
            sigma_i = self.grid_search(X[i] - X, i, perplexity)
            norm = np.linalg.norm(diff, axis=1)
            p_i_given_j[i,:] = np.exp(-norm**2/(2*sigma_i**2))
           # p_i_given_j[i, :] = np.exp(-(X[i] - X) ** 2 / (2 * sigma_i ** 2))
            np.fill_diagonal(p_i_given_j, 0)
            p_i_given_j[i, :] /= np.sum(p_i_given_j[i, :]) + 1e-7
        return p_i_given_j

    def grid_search(self,diff_i,current,perplexity):
        norm = np.linalg.norm(diff_i, axis=1)
        result = np.inf
        norm = np.linalg.norm(diff_i, axis=1)
        std_norm = np.std(norm)
        for sigma_search in np.linspace(0.01*std_norm,5*std_norm,500):
            p = np.exp(-norm ** 2 / (2 * sigma_search ** 2))
            p[current] = 0
            epsilon = 1e-7
            p_new = np.maximum(p/np.sum(p), epsilon)
            H = -np.sum(p_new * np.log2(p_new))

            if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
                result = np.log(perplexity) - H * np.log(2)
                sigma = sigma_search
        return sigma

    def make_symmetric(self,p_i_given_j):
        n = len(p_i_given_j)
        p_ij_symmetric = np.zeros(shape=(n, n))
        for i in range(0, n):
            for j in range(0, n):
                p_ij_symmetric[i, j] = (p_i_given_j[i, j] + p_i_given_j[j, i]) / (2 * n)
        e = np.nextafter(0, 1)
        p_ij_symmetric = np.maximum(p_ij_symmetric, e)
        return p_ij_symmetric

    def initilization(self,X):
        #initializing Y
        return np.random.normal(loc=0, scale=1e-4, size=(len(X), 2))

    def low_dimensional_matrix(self,Y):
        n = len(Y)
        q_i_given_j = np.zeros((n, n))
        for i in range(n):
            diff = Y[i] - Y
            norm = np.linalg.norm(diff, axis=1)
            q_i_given_j[i, :] = (1 + norm ** 2) ** (-1)
        np.fill_diagonal(q_i_given_j, 0)
        q_i_given_j /= q_i_given_j.sum()
        e = np.nextafter(0, 1)
        q_i_given_j = np.maximum(q_i_given_j, e)

        return q_i_given_j

    def gradient_descent(self,Y,p_i_given_j,q_i_given_j):
        n = len(p_i_given_j)
        gradient = np.zeros((n, 2))
        for i in range(n):
            diff = Y[i] - Y
            A = np.array([p_i_given_j[i,:] - q_i_given_j[i,:]])
            B = np.array([(1 + np.linalg.norm(Y[i] - Y, axis=1) ** 2) ** -1])
            C = diff
            gradient[i] = 4 * np.sum((A * B).T * C, axis=0)
        return gradient

    def tSNE(self,X,perplexity=10,T=500,lr=200):
        n = len(X)
        p_ij = self.compute_pairwise_affinities(X,perplexity)
        p_ij = self.make_symmetric(p_ij)
        Y = np.zeros(shape=(T, n, 2))
        Y[0] = np.zeros(shape=(n, 2))
        Y_1 = self.initilization(X)
        Y[1] = np.array(Y_1)
        for t in range(1, T-1):
            q_ij = self.low_dimensional_matrix(Y[t])
            gradient = self.gradient_descent(Y[t],p_ij,q_ij)
            Y[t+1] = Y[t] - lr * gradient + 0.9 * (Y[t] - Y[t-1])
            if t % 50 == 0 or t == 1:
                cost = np.sum(p_ij * np.log(p_ij / q_ij + 1e-7))
                print(f"Iteration {t}: Value of Cost Function is {cost}")
        solution = Y[-1]
        return solution,Y




if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    import pandas as pd

    # Fetch MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.uint8)

    X_total = pd.DataFrame(mnist["data"])
    y_total = pd.DataFrame(mnist["target"])

    X_reduced = X_total.sample(n=1000)
    y_reduced = y_total.loc[X_reduced.index]
    # print(X_reduced.shape)
    # print(y_reduced.shape)
    # # PCA to keep 30 components
    X = PCA(n_components=30).fit_transform(X_reduced)

    tsne = t_SNE(X)
    solution,Y = tsne.tSNE(X,perplexity=30,T=1000,lr=200)
    # print(solution)
    # print(Y)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(solution[:, 0], solution[:, 1], c=y_reduced.values.ravel(), cmap="jet")
    plt.title("t-SNE")
    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=y_reduced.values.ravel(), cmap="jet")
    plt.title("PCA")
    plt.show()
    # TSNE = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    # X_embedded = TSNE.fit_transform(X)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_reduced.values.ravel(), cmap="jet")
    # plt.title("t-SNE")
    # plt.subplot(122)
    # plt.scatter(X[:, 0], X[:, 1], c=y_reduced.values.ravel(), cmap="jet")
    # plt.title("PCA")
    # plt.show()

