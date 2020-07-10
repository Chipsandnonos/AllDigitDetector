import nueron
import random
import numpy as np

class Layer( ) :
    def __init__(self, n_laysize, prev_laysize):

        self.n_laysize = n_laysize
        self.prev_laysize = prev_laysize
        self.neurons = []
        self.weights = []
        for i in range(self.n_laysize):
            self.neurons.append(nueron.Neuron())
        #self.neurons = np.array(self.neurons).T

        if prev_laysize != 0:
            for n in range(self.n_laysize):
                row = []
                for i in range(prev_laysize):
                    row.append(random.random())
                self.weights.append(row)
            self.weights = np.array(self.weights)

    def collect_activations(self):
        activations = []
        for i in range(self.n_laysize):
            activations.append(self.neurons[i].activation)
        return activations

    def update_activations(self, activations):
        if self.prev_laysize != 0:

            updated = np.dot(self.weights, activations)

            for i in range(self.n_laysize):
                self.neurons[i].activation = self.sigmoid(updated[i])
        else:
            for i in range(self.n_laysize):
                self.neurons[i].activation = activations[i]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # This is the derivative of the sigmoid curve
    def sigmoid_p(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))









