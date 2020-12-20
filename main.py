from fcm import FCM
import pandas as pd

coordinates = pd.read_csv("sample1.csv").values
m = 1.5
termination_condition_threshold = 0.01
second_last_entropy = None
last_entropy = None
entropy = None
minimum_entropy = None
best_number_of_clusters = None
number_of_clusters = 2

while second_last_entropy is None or entropy < last_entropy or last_entropy < second_last_entropy:
    print("Trying {} Clusters...".format(number_of_clusters))
    fcm = FCM(coordinates, number_of_clusters, m)
    second_last_entropy = last_entropy
    last_entropy = entropy
    entropy = fcm.start(termination_condition_threshold)
    if minimum_entropy is None or entropy < minimum_entropy:
        best_number_of_clusters = number_of_clusters
        minimum_entropy = entropy
    number_of_clusters += 1
print()

print("Best Number of Clusters: {}".format(best_number_of_clusters))
fcm = FCM(coordinates, best_number_of_clusters, m)
entropy = fcm.start(termination_condition_threshold)
print("Entropy: {}".format(entropy))

# Only For 2-Dimensional Data
fcm.plot_data()
fcm.plot_boundaries()
