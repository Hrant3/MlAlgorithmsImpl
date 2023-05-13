import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

class SpectralClustering():
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters

    def epsilon_affinity(self, X, epsilon=1):
        m = X.shape[0]
        W = np.zeros((m, m))
        D = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if np.linalg.norm(X[i] - X[j]) <= epsilon:
                    W[i, j] = 1
                    D[i, i] += 1
        return W, D


    def un_normalized_laplacian(self,X,num_eigenvectors):
        W,D = self.epsilon_affinity(X,epsilon=2)
        L = D - W
        _, U = eigh(L,subset_by_index=(0,num_eigenvectors-1))
        return U

    def predict(self,X):
        U = self.un_normalized_laplacian(X,self.n_clusters)
        clusters = KMeans(n_clusters=self.n_clusters).fit(U).labels_
        return clusters



def main():
    # X, y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)
    # clusters = SpectralClustering(2).predict(X)
    # plt.scatter(X[:,0], X[:,1], c=clusters, cmap='Paired')
    # plt.show()
    X, y = datasets.make_blobs(n_samples=300, random_state=170)
    clusters = SpectralClustering(3).predict(X)
    print(clusters)
    plt.scatter(X[:,0], X[:,1], c=clusters, cmap='Paired')
    plt.show()

main()

