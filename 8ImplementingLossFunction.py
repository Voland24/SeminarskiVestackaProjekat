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
        t = np.linspace(class_number*4,(class_number+1)*4,points) + np.random.randn(points)*1
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    intValy = []    
    for val in y:
        intValy.append(int(val))

    npArrayType = np.array(intValy)
    return X,npArrayType

class Loss:
    def calculate(self, output, y): #output is output from the neural network output layer, y are the target values
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)  #the mean loss for this entire batch of output values for given batch samples
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_prediction, y_true):
        samples = len(y_prediction)
        y_prediction_clipped = np.clip(y_prediction,1e-7,1-1e-7) #

        #problem: y_true can be either [1,0] or they can be one hot [[0,1],[1,0]]
        #we solve for both, check and adjust

        if len(y_true.shape) == 1:  #that means [1,0] was passed, as a scalar array (1,n)
            correct_confidence = y_prediction_clipped[range(samples), y_true] #we're taking all of the rows of the predicted_clipped and taking those we are checking, because the y_true is [0,1] so the [0,0] element and the [1,1] element for example
        elif len(y_true.shape) == 2: #one hot encoding
            correct_confidence = np.sum(y_prediction_clipped*y_true, axis=1)    #becasuse of axis=1, we mul the clipped predictions rows by rows of y_true which is now one hot encoded, and when we sum, we sum by rows as well, so we get the same shape as in the first if statement

        # correct confidences looks like [a,b,c]T now, it's just a column of values
        negative_log_likelihoods = -np.log(correct_confidence) #calculate cross entropy loss
        return negative_log_likelihoods    


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

loss_fucntion = Loss_CategoricalCrossEntropy()
loss = loss_fucntion.calculate(activation2.output, y)

print("Loss: ", loss)

predictions = np.argmax(activation2.output, axis=1)

accuracy = np.mean(predictions == y)

print("Accuracy: ", accuracy)

