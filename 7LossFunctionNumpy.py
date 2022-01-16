
#now we need to make it work with batch outputs of the network

#for example

#outputs are [[], [], []] for 3 different samples
#the target classes are 0,1,2 coded
#and we're not using one hot encoding
#and we have [0,1,1] as the target predictions
#meaning the first sample is supposed to predict the class 1, that probability should be the highest
#the second sample was of class 2 and the output for the second sample is supposed to predict the class 2 etc

import numpy as np

softmax_output = np.array([ [0.7,0.1,0.2], [0.1,0.5,0.4], [0.02,0.9,0.08] ])

class_targets = [0,1,1]

print(softmax_output[[0,1,2], class_targets])

#because of numpy, we can pass [0,1,2] meaning these are the first dimension indices we're interested in
#or put simply, take these rows and on each get me the class_targets_batch indices
#so [0,1,2], [0,1,1] means take the elements m[0,0] m[1,1] m[2,1]

#easier way is to instead of saying [0,1,2] for the rows, 

print(softmax_output[range(len(softmax_output)), class_targets])

#and to apply loss fucntion calculation we simply do

print(-np.log(softmax_output[range(len(softmax_output)), class_targets]))


#there is a problem with a prediction being 0, beacuse the log(0) is -inf
#we solve this by clipping the values by something very very small, like  1e-7

