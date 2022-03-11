import mnist_loader
import network2
import json

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 30, 10,10])#change the number of layers or number of neurons in each layer here
validation_data = list(validation_data)
training_data = list(training_data)
#(training data, epoch, LR, ??
"""
#from chapter 3 of the book
training_data is a list of tuples (x,y) => training input/expected output
epoch - training iteration
mini_batch size - size of batch segment for training
eta - learning_rate (? eta)
lmbda - regularization parameter
remainder are optional parameters that should be self explanatory
"""
net.SGD(training_data, 1000, 50, 0.08, lmbda=2.5,evaluation_data=validation_data, monitor_evaluation_accuracy=True)
net.save("WeigntsAndBiases.txt")