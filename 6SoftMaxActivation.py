#activation function used for the output layer of the neural network

layer_output = [4.8,1.21,2.385]

layer_output2 = [4.8,4.79,4.25]

#both of these outputs are accuracy wise, identical, they have the same 4.8 values as the highest one
#however, which one was less wrong, which one was better

#especially if the output values are unbouned, the reLU just says x if x > 0, so the difference
#between output values can be bigger between sample data given

#and now we need SoftMax activation function to do this

#what is this exactly? Well we want out output values from the network, the output layer values to
#be probability distributions, or better said, that their sum is 1, it gives out the probability of 
#that output layer's neuron's guess to be correct

#and now the values are normalized and it's so much easier to compare
#the 100% correct prediction would just be one neuron giving 1, and the rest just give 0

#instead of doing nothing with the output layer's output values, or squaring them or doing an abs fucntion
#we can use exponential functions to determine what to do with these values and make them suitable for 
#backpropagation

# f(x) = e exp(x)

# when x > 0, we can e^x, when x <= 0, we get e^(-x) or 1 / e^x which is still not negative
#and it didn't ignore the negative value of the x param, meaning it gives meaning to negative values
#while keeping all of the output values positive

import math
import numpy as np

layer_output = [4.8,1.21,2.385]

E = math.e

exp_values = []

for output in layer_output:
    exp_values.append(E**output)  #this applies exponential function to our values


exp_values_np = np.exp(layer_output)  #numpy way of doing this

#now we need to normalize the output values
# 
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value/ norm_base)   #values divided by the sum of all values


norm_values_np = exp_values_np / np.sum(exp_values) #this is a quick way of doing this    

print(sum(norm_values_np))  #this is going to be 1    
#


#ova kombinacija exp i normalize is actually the SoftMax activation function


#Make this work with batch outputs, because we'll have batch inputs

layer_output_batch = [[4.8,1.21,2.385],[8.9,-1.81,0.2],[1.41,1.051,0.026]]

exp_values_batch = np.exp(layer_output_batch) #np works with batch data immediately

norm_value_batch = exp_values_batch / np.sum(exp_values_batch, axis=1, keepdims=True)

#axis 1 means that it will add the values in a single row, so we can get (1,3) shape
#keepDims is used because the exp_values_batch is (3,3) shape, we want the sum to be of
# (3,1) shape, and keepDims will do that for us


#the only problem left is that before doing the exp, we can get overflow error because the e^x can be really big
# that's why we clean that data by substracting the max(input) from every memmber, so that the largest one is 0
#that just means the biggest number we can get by e^x is 0, because the greatest x is going to be zero

