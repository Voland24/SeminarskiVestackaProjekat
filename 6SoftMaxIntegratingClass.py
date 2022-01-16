import numpy as np

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) #scales in by 0.1 fo magnitude, creates an inputs*neurons matrix populated with random numbers, like this, we don't need the transpose later
        self.biases = np.zeros((1,n_neurons))  #creates a 1*neurons matrix full of zeros                                                         
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases   


class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class SoftMaxActivation:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #keeping exp low
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True) #calculating the probs
        self.output = probabilities        

def create_data(points,classes):  #how many dots you want, how many classes you want
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes)
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0,1,points)
        t = np.linspace(class_number*4,(class_number+1)*4,points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X,y


X,y = create_data(100,3)

dense1 = Layer_Dense(2,3)  #2 input features, 3 neurons
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)  # 3 outputs from previous layer, and because we have 3 classes in the data, 3 outputs from this, the output layer
activation2 = SoftMaxActivation()

dense1.forward(X)  #the entry data
activation1.forward(dense1.output)  #going throw reLU activation

dense2.forward(activation1.output)  #the previous layer's output is the input here
activation2.forward(dense2.output)  #going through the softmax activation and getting predictions

print(activation2.output[:5])



