import numpy as np

np.random.seed(0)

#inputs are called X
X = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]


#we're going to make hidden layers, that just means, we're going to make a layer into an object
#we're not going to manually change it, the network in it's flow will do it for us

#save a model means "save the weights and biases"
#when we're init the weights, we make them somewhere in the range of -0.1,0.1, the smaller the better
#and over time the network will tune these for us

#we have to normalize and scale the inputs as well, make them all in the -1,1 range

#usually biases are init to 0, but not always, because sometimes the input*weight is small enough
#so the neuron doesn't activate, and if the bias is 0, the network is dead

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) #scales in by 0.1 fo magnitude, creates an inputs*neurons matrix populated with random numbers, like this, we don't need the transpose later
        self.biases = np.zeros((1,n_neurons))  #creates a 1*neurons matrix full of zeros                                                         
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases   


layer1 = Layer_Dense(4,5)  # howManyFeaturesInOneSample * how many neurons we want in this layer
layer2 = Layer_Dense(5,5)  # how many neurons were in the previous layer's output * how many neurons for this layer

layer1.forward(X)
print('Layer 1 output')
print(layer1.output)
print('\n')
print('Layer 2 output')
layer2.forward(layer1.output)
print(layer2.output)