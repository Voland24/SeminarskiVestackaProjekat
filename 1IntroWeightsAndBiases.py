inputs = [1,2,3,2.5]  ##data coming from previous neurons
 
weights = [0.2,0.8,-0.5,1.0]  ##weights on those connections from previous neurons

bias = 2 ##bias of the neuron we're coding

output = inputs[0]*weights[0] + inputs[1]*weights[1] +inputs[2]*weights[2] + inputs[3]*weights[3] + bias

##output of a neuron is the dot product of weights and inputs of previous neurons, those inputs are the outputs of
##each previous neruon
##and we add the bias at the end


##the inputs can either be coming from the first layer, the input layer or it can be the outputs of the previous neurons

##the input layer are the variables that we're tracking
## heat sensors, humidity, air, ... info 


##if we want the output layer for example, we can have 3 neurons in that layer with 4 inputs each i.e. there are 4
##neurons in the previous layer
## so we have the inputs array that stays the same, 4 different input values from 4 neurons
##but we need 3 different weights arrays of 4 elements each, 3 because we're modelling 3 neurons with 4 connections each
##from the previous neurons
##and we need 3 separate biases, one for each neuron

weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5


output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] +inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] +inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] +inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]


##how do we change or control the outputs of the neurons? We change either the bias of the neuron or by
##changing the weights of the connections, and thus we change we output of the neuron we are trying to control

print(output)

