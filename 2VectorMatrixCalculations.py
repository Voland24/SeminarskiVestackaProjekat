

inputs = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]

biases = [ 2,3,0.5]

layer_output = [] ##output of given layer
for neuron_weights, neuron_bias in zip(weights,biases):  #we have [[weightForGivenNeuron], biasofGivenNeuron] as element
    neuron_output = 0
    for n_input, weight in zip(inputs,neuron_weights): ##we have [inputOfPreviousNeuron,WeightFromThatNeuron]
        neuron_output +=n_input*weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)


##biases and weights are tuneable elements of a neural network
##we don't actually need both, however it gives more tuneable parameters if we use both

##biases offset the input value, weights apmlify it's impact, or dimish it if the value is negative, 
##the weights are changing the magnitude of the input

print(layer_output)        