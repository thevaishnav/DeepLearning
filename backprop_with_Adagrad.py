import numpy as np
import random


def AF(Z):
    # Activation Function (Sigmoid)
    Z = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z))


def prime(Z):
    # return derivative of Z with respect to Weights
    sig = AF(Z)
    return sig * (1 - sig)


class Network:
    def __init__(self, layers):
        """
        :param layers: a list containing number of neurons per layer for each layer. (including input and output layer)
        """
        self.layers = layers
        self.activations = []
        self.Zs = []
        self.init_parameters()

    def init_parameters(self):
        self.weights = [np.random.randn(i, j) for i, j in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [np.random.randn(1, i) for i in self.layers[1:]]
        self.SWs = [0, ] * len(self.weights)
        self.SBs = [0, ] * len(self.biases)

    def feed_forward(self, X):
        """
        :param X: numpy array.
        :return: activations of last layer.
        """
        self.activations = [X]
        self.Zs = []
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = AF(z)
            self.activations.append(a)
            self.Zs.append(z)
        return a

    def back_prop(self, X, Y, batch_size, eta=0.005, epsilon=0.00005, epochs=5, epoch_complete_call=None):
        """
        :param X: input values (Numpy array)
        :param Y: output values (Numpy array)
        :param batch_size: integer (how many training examples per batch)
        :param eta: Learning rate
        :param epochs: number of epochs to perform.
        :param epoch_complete_call: A function that should be called after each epoch.
        """
        training_data = list(zip(X, Y))

        for epch in range(epochs):
            random.shuffle(training_data)
            X, Y = zip(*training_data)
            mini_batches = [(X[k - batch_size:k], Y[k - batch_size:k])
                            for k in range(batch_size, len(training_data), batch_size)]
            for minix, miniy in mini_batches:
                batchX, batchY = np.array(list(minix)), np.array(list(miniy))
                self.feed_forward(batchX)
                self.update_weights(batchY, eta, epsilon)

            if epoch_complete_call is not None:
                epoch_complete_call(epch)

    def update_weights(self, batchY, eta, epsilon):
        """internal function"""
        deltas = [self.activations[-1] - batchY]
        delta = deltas[0]
        deltaWs = [eta * np.dot(self.activations[-2].T, delta)]
        deltaBs = [eta * np.mean(delta, axis=0)]

        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * prime(self.Zs[i - 1])
            deltas.insert(0, delta)
            deltaWs.insert(0, eta * np.dot(self.activations[i - 1].T, delta))
            deltaBs.insert(0, eta * np.mean(delta, axis=0))

        for i in range(len(self.weights)):
            self.SWs[i] = self.SWs[i] + (deltaWs[i] ** 2)
            self.SBs[i] = self.SBs[i] + (deltaBs[i] ** 2)
            self.weights[i] -= eta * deltaWs[i] / ((self.SWs[i] ** 0.5) + epsilon)
            self.biases[i] -= eta * deltaBs[i] / ((self.SBs[i] ** 0.5) + epsilon)
