#batch of inputs, the inputs are going to be a 2d array or a matrix

#most of these are done on GPU's, more cores, more parallel
#we want generalization, more inputs, means more variables impacting the decision
#i.e more sensor data at different times of day, each row would be one reading of samples

#we give the machine more sample data at once, more chances the functions guesses properly

#giving it one sample at a time is a horrible way to predict, the function would fluctuate more
#also, we're not giving it all sample data at once, we're trying to find the best sample giving rate
#for the best performance and fitting, best ones are 32,64 at once

#the weights and biases stay the same in terms of shape. Why?
#we still have 3 neurons with three biases and 4 connections to each of them, that's why
#the input matrix is (3,4), the 4 is because of the connections
#we just have more sample data to give it, in terms of different times of day
#the samples were taken for example



#weights are (3,4) * inputs(3,4) won't work
#we're going to transpose weight and do inputs(3,4) * weightsT(4,3) and get 3x3 matrix of outputs
import numpy as np

inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]

biases = [ 2,3,0.5]

#adding a new layer
#meaning we have new weights connecting them and new biases for those new neurons we're adding

#why are awight2 (3,3), well we have 3 neurons in the previous layer, and 3 here
#the first three is because we have 3 neurons in this new layer
#the second three is because the previous layer has 3 neurons so there are going to be
#three new lines to each neuron in the new layer

weights2 = [[0.1,-0.14,0.5],[-0.5,0.12,-0.33],[-0.44,0.73,-0.13]]

biases2 = [-1,2,-0.5]

weights = np.array(weights).T  #transpose , weights are now (4,3) matrix 

layer1_outputs = np.dot(inputs, weights) + biases

#these are the inputs into the new layer

#layer1_outputs are (3,3), weights2 are 3x3 as well, however to multiply the right values by the right ones,
#we have to transpose

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
