import layer
from mnist import MNIST
import numpy as np
import random

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
        self.index += 1


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

    def learn(self):
       self.run_entry()
       actual = []
       for i in range (10):
           if (i == self.index):
               actual.append(i)
           else:
               actual.append(0)
       costvector = []

       for i in range (10):
           cost = (self.layer4[i] - actual[i])**2
           costvector.append(cost)





        #get all the vectors for n training example

        #averages
        #applies to the actual weights and biases
        

#weird that its all .99 ?
nn = NueralNetwork()
print(nn.train_labels[1], type(nn.train_labels[1]))
#nn.run_entry()

