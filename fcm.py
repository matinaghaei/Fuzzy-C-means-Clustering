from clusters import Clusters
from data import Data
import matplotlib.pyplot as plt
import numpy as np


class FCM:

    def __init__(self, coordinates, number_of_clusters, m):
        self.m = m
        self.clusters = Clusters(number_of_clusters, coordinates.shape[1])
        self.data = Data(coordinates, self.clusters, m)
        self.clusters.set_data(self.data)

    def start(self, termination_condition_threshold):
        while self.data.update_memberships() >= termination_condition_threshold:
            self.clusters.update_centers()
        return self.data.get_entropy()

    def plot_data(self):
        self.data.plot()
        self.clusters.plot()
        plt.show()

    def plot_boundaries(self):
        x = np.arange(0, 1, 0.03)
        y = np.arange(0, 1, 0.03)
        plot_coordinates = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        plot_data = Data(plot_coordinates, self.clusters, self.m)
        plot_data.update_memberships()
        plot_data.plot()
        plt.show()
