# DeepLearning
***
This project by Vaishnav Chincholkar is an implementation of Deep Learning algorythm in Raw Python using Numpy.
Run "test_example.py" file with Python 3.
The program should ask for otimizer, which can be any of the following string:
  * RMSProp
  * Adagrad 
  * Adam
  * Momentum
 
 Each corrusponding to one of the algorythm.
 
To use the Network in individual project, create a Network class variable which would need an input parameter called layers. This parameter can be any type of itterator.
This parameter will deside the number of neurons in each of the layer in netowrk as well as number of layers in the netowrk.
Each "int" element in the layers parameter will be interprited as number of neurons in the currosponding layer, for example if the parameter is
  
  [100, 80, 50, 10]

This means that the input layer has 100 neurons, 1st hidden layer has 80 neurons, 2nd hidden layer has 50, and output layer will have 10 neurons.


To get the prediction from network for a perticular input call Network.feed_forward(X) function. Where X is the input for which prediction is to be made.
To train the model on a perticular dataset call Network.back_prop(X, Y, batch_size, eta=0.005, epoch_complete_call=None)
Parameters for Network.back_prop functions are:
  * X = Training Input
  * Y = Training Labels
  * batch_size = Number of samples in each training batch
  * eta = Learning Rate
  * epoch_complete_call = This is he function which will be called after every epoch to test the accuracy of network
  
 
 Network will also have other optional parameters which will each currospond to the hyperparameter for that method.
