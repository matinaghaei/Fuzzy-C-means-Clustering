import math
import numpy as np
import matplotlib.pyplot as plt


class Data:
    colors = ['green', 'yellow', 'blue', 'magenta']

    @staticmethod
    def cal_distance(coordinate1, coordinate2):
        s = 0
        for d in range(len(coordinate1)):
            s += (coordinate1[d] - coordinate2[d]) ** 2
        return math.sqrt(s)

    def __init__(self, coordinates, clusters, m):
        self.coordinates = coordinates
        self.clusters = clusters
        self.m = m
        self.memberships = np.zeros((len(coordinates), clusters.size()))

    def size(self):
        return len(self.coordinates)

    def get_membership(self, data_index, cluster_index):
        return self.memberships[data_index, cluster_index]

    def get_data(self, data_index):
        return self.coordinates[data_index]

    def update_memberships(self):
        new_memberships = np.zeros((len(self.coordinates), self.clusters.size()))
        for i in range(len(self.coordinates)):
            for j in range(self.clusters.size()):
                s = 0
                for k in range(self.clusters.size()):
                    d1 = self.cal_distance(self.coordinates[i], self.clusters.get_center(j))
                    d2 = self.cal_distance(self.coordinates[i], self.clusters.get_center(k))
                    s += (d1 / d2) ** (2 / (self.m - 1))
                new_memberships[i, j] = 1 / s
        diff = np.amax(np.absolute(new_memberships - self.memberships))
        self.memberships = new_memberships
        return diff

    def plot(self):
        for i in range(len(self.coordinates)):
            plt.scatter(self.coordinates[i, 0], self.coordinates[i, 1], c=self.colors[self.get_cluster(i)])

    def get_cluster(self, index):
        maximum_membership = 0
        cluster = 0
        for j in range(self.clusters.size()):
            if self.memberships[index, j] > maximum_membership:
                maximum_membership = self.memberships[index, j]
                cluster = j
        return cluster

    def get_entropy(self):
        s = 0
        for i in range(len(self.coordinates)):
            for j in range(self.clusters.size()):
                s += self.memberships[i][j] * math.log(self.memberships[i][j])
        return -s
