import numpy as np
import csv

class Searcher(object):
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def search(self, queryFeatures, limit=10):
        distance = {}

        # Open feature mapping file
        with open(self.indexPath) as f:
            reader = csv.reader(f)

            for row in reader:
                # Grab feature vector from csv
                features = [float(x) for x in row[1:]]
                # Compute matrix
                d = self.search_distance(features, queryFeatures)

                distance[row[0]] = d

        f.close()

        # Sort the distance matrix so the most relevant image will be at the front of list
        distance = sorted([(value, key) for (key, value) in distance.items()])
        return distance[:limit]


    def search_distance(self, vector1, vector2):
        # Using chi-square distance here, plus 1e-10 is for preventing potential divded by zero error.
        d = 0.5 * np.sum([((a-b) ** 2) / (a+b+1e-10) for (a, b) in zip(vector1, vector2)])
        return d
