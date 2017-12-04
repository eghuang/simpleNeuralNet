import numpy as np

import backend

class Perceptron(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.get_data_and_monitor = backend.make_get_data_and_monitor_perceptron()
        self.weights = np.zeros(dimensions)

    def get_weights(self):
        """
        Return the current weights of the perceptron.

        Returns: a numpy array with D elements, where D is the value of the
            `dimensions` parameter passed to Perceptron.__init__
        """
        return self.weights

    def predict(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if np.dot(x, self.get_weights()) >= 0:
            return 1
        return -1

    def update(self, x, y):
        """
        Update the weights of the perceptron based on a single example.
            x is a numpy array with D elements, where D is the value of the
                `dimensions`  parameter passed to Perceptron.__init__
            y is either 1 or -1

        Returns:
            True if the perceptron weights have changed, False otherwise
        """
        if self.predict(x) == y:
            return False
        self.weights += y * x
        return True

    def train(self):
        """
        Train the perceptron until convergence.
        """
        def tryPass():
            result = True
            for x, y in self.get_data_and_monitor(self):
                if self.update(x, y) == True:
                    result = False
            return result

        while tryPass() == False:
            continue
