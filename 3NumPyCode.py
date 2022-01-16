#list is a 1d array or a Vector in maths

# [1,2,3,4] has a shape of (1,4) 1 row, 4 columns
# [[1,2,3,4],[5,6,7,8]] has a shape of (2,4) and it's a 2d array i.e a matrix

#we can't have jaged matricies, i.e [[1],[2,3,5],[2,5]] we can't tell it's shape

# [
#  [[1,2,3,4],[4,5,6,7]],
#  [[],[]],
#  [[],[]]
# ]

#this has a shape of (3,2,4)
# it has 3 outmost elements, each of which has 2 sublists, each of which has 4 elements or integers
# this is a 3D object



#tensor is an object that can be represented as an array

import numpy as np


inputs = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]

biases = [ 2,3,0.5]

output = np.dot(weights, inputs) + biases

print(output)

#weights (3,4) is mul by inputs(1,4) so it can be mulled
#dot is elements wise mul and add
#again, each weight i.e connection from previous neuron is mulled by the output of that neuron
#or better said the input into this neuron and just adding the bias at the end

