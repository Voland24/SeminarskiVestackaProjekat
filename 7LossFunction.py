#we need a metric for errors of the output layer neurons

#but we don't use the accuracy for the grading method of each neurons performance
#doing this, we throw away a lot of data
#the network actually outputs the probablity distribution for each of the classes in the given dataset

#it predicts class 1 with 99% confidence (i.e the neuron for that class outputs 0.99) and in the other sample
#from the batch it gives a 0.335% confidence. It still true i.e correct because that's the greatest score, but how close did it come to not be correct


#we need a Loss function
#we need to see how wrong we were on the predictions given the output for a known target i.e training data

#if we're using SoftMax Activation for the output on a classification problem we would use
#Categorical Cross Entropy Loss Function

#we're giving the function 2 probability distributions
#we're taking the negative sum of the product of the target value and the log of the predicted value, for each target we have

#One Hot Encoding

#we have a vector that (1,n) where n is the number of classes we have to predict
#the vector is all zeros except on the index of the target class label 
#i.e we would have [1,0,0] for label = 0 for 3 classes

import math
import numpy as np
softmax_output = [0.7,0.1,0.2]  #the network's output on the output layer

target_class = 0 #this means the input given was of class 0, the network's supposed to predict this

target_output = [1,0,0] #i.e the one hot vector, this is how the distribution is supposed to look like
                        #for this given target class

#we can simplify the categorical cross entropy function 
#because using the one hot vector, we have 0*something
#everywhere except where the target_output is 1
#and we use that to bring and see how wrong
#the correct predicition was here
#i.e how far off the prediction for that label
#is in this output 
loss = -(math.log(softmax_output[0])*target_output[0] +
        math.log(softmax_output[1])*target_output[1] + #this product would zero out
        math.log(softmax_output[2])*target_output[2])  #so would this one

loss_same = -math.log(softmax_output[0])*target_output[0]  #this is the same as the upper calculation of loss         


print(loss)





