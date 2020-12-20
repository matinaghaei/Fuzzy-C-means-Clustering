import numpy as np
import matplotlib.pyplot as plt


class Clusters:

    def __init__(self, number_of_clusters, dims):
        self.number_of_clusters = number_of_clusters
        self.dims = dims
        self.centers = np.random.rand(number_of_clusters, dims)
        self.data = None

    def set_data(self, data):
        self.data = data

    def size(self):
        return self.number_of_clusters

    def get_center(self, index):
        return self.centers[index]

    def update_centers(self):
        for j in range(self.number_of_clusters):
            s1 = np.zeros(self.dims)
            s2 = 0
            for i in range(self.data.size()):
                s1 += self.data.get_membership(i, j) * self.data.get_data(i)
                s2 += self.data.get_membership(i, j)
            self.centers[j] = s1 / s2

    def plot(self):
        for center in self.centers:
            plt.scatter(center[0], center[1], s=100, c='red')
