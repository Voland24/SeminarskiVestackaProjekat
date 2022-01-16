#Step function
#it gives 1, for x > 0 and 0 for x <=0
#determines wheter or not the neuron will fire off, the neurons will give either a 1 or 0 off
#1 or 0 will be the output for each neuron

#Sigmoid activation function
# f(x) = 1/1+e exp(-1)
#granular output
#after you do inputs*weights + bias, we give that to the activation function and it determines wheter or not it
#fires off
#later, when we optimize the netowrk i.e. tweak the weights and biases, we need to see how much loss
#i.e how wrong the network was on some guesses, and that's easier with the sigmoid function

#Rectified linear unit
# it's like the step function but
# f(x) is x, for x >0 and 0 for x <=0
#it's also granular and very fast, because it's simple
#it just works, guys



#without the activation function, we can't model non-linear functions 
#when we don't use any activation functions, we still use one, it's just f(x) = x, we jsut pass values along
#and that's a linear function and it's not good for modelling


#with reLU, the bias determines when the neuron fires off or stops firing off
#the weights, when positive, say that the neuron will fire off wherever the bias told it to
#and when negative, tells the neuron to stop firing off wherever the bias told it too
#this is all for one single neuron

#if we add another neuron, it's bias tilts the output of the first one vertically, when the weights are positive
#simillarly, if we tweak the weights of the second neuron so that they're negative,
#we give the lower and upper bound of activation for the neurons
#if both neurons are activated, the output of the second one is in some range defined by the weights

#if the first neuron isn't activated, it will give out 0, and the output of the second neuron will just be it's bias
#and if the second neuron isn't activated, then the output will be just 0, regardless of the first neuron

#in order to fit non-linear problems with neural networks, we need atleast 2 hidden layers, and use a
#non linear activation function



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
        self.output = np.maximum(0,inputs) # if the val is <0, 0 is max, else val is max, this is rectified linear unit


#creates 3 spirals od dots coming from a single point of origin
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

#we can create 3 different classes of data, each with points amount of representatives, each of which has
#only two features in them, i.e coordinates on the graph

X,y = create_data(100,3)   

layer1 = Layer_Dense(2,5)  #features now are just coords of the point, so we have 2 features, 5 neurons is just a choice

activation1 = Activation_ReLU()

layer1.forward(X)  #creates outputs from layer1

activation1.forward(layer1.output) # does the activation function for layer1 outputs

print(activation1.output)   #how each of the output values from layer1, makes all the negative values from
                            #layer1 into 0 values, and keeps the positive ones




