from __future__ import print_function, division
import numpy as np
import math

def squared_euclidean_distance(x1, x2):
    distance = 0
    if len(x1) != len(x2):
        raise ValueError('The points must have the same dimension!')
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

def manhattan_distance(x1, x2):
    distance = 0
    if len(x1) != len(x2):
        raise ValueError('The points must have the same dimension!')
    for i in range(len(x1)):
        distance += np.abs(x1[i] - x2[i])
    return distance

class KNN():
    """ K Nearest Neighbors Classifier.

    Parameters:
    ----------
    k: int
        The number of closest neighbors that will determine the class of the sample that we wish to predict
    """
    def __init__(self, k=5):
        self.k = k

    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples"""
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()
    
    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([squared_euclidean_distance(test_sample, x) for x in X_train])[:self.k] # [:self.k] slices the list to keep only the indices of the k closest training samples
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred