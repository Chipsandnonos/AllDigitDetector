import nueron
import random

class Layer( ) :
    def __init__(self, n_laysize, n_sub_1_laysize ):
        self.n_laysize = n_laysize
        self.n_sub_1_laysize = n_sub_1_laysize
        self.neurons = []
        self.weights = []
        for i in range(self.n_laysize):
            self.neurons.append(nueron.Neuron())

        for n in range(self.n_laysize):
            row = []
            for i in range(n_sub_1_laysize):
                row.append(random.random())
            self.weights.append(row)

    





