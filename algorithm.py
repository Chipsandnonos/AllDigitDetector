import layer
from mnist import MNIST
import numpy as np


class NueralNetwork():

    def __init__(self):
        self.layers = []
        self.layer1 = layer.Layer(28**2, 0)
        self.layer2 = layer.Layer(16,28**2)
        self.layer3 = layer.Layer(16,16)
        self.layer4 = layer.Layer(10,16)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)
        self.mndata = MNIST('samples')
        self.train_images, self.train_labels = self.mndata.load_training()
        self.index = 0

    def run_entry(self): #make sure to increment index
        activations = self.translate()

        for i in range (len(self.layers)):

            self.layers[i].update_activations(activations)
            activations = self.layers[i].collect_activations()


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # This is the derivative of the sigmoid curve
    def sigmoid_p(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def translate(self):
        activations = []
        for y in range (28):
            for x in range (28):
                activations.append(self.sigmoid(self.train_images[self.index][x + y*28]))
        return activations

#weird that its all .99 ?
nn = NueralNetwork()
nn.run_entry()