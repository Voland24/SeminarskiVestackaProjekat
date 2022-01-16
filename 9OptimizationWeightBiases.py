
#we can but shouldn't randomly choose weights and biases for the network
#we will decrease loss, but slightly, and the accuracy would barely improve, if at all

#we can randomly tweak the best known weights and biases, i.e when we determine that the
#current combination of weights and biases has decreased lost, we keep it as a current
#and randomly tweak the weights and biases of it to see if it will further decrease loss

#however, this will work for only linear problems and doesn't scale well for non linear data

#we must also find which weights and biases influence the model more and we must avoid local minimum

#impact of a variable, i.e impact of weights and models on the loss function for each of the samples
#and weights and biases impact the model output either way

#the influence on x on y for f(x) = 2x is just 2 because it's a linear function

#for non linear functions we need derivatives i.e the first derivative

#we use the slope, but of the tangent line, infinitely close points

#we wanna use two very very close points