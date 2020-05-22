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
        self.init_weights()
        self.init_biases()

    def init_weights(self):
        """initialize weights randomly with normal distribution 1"""
        self.weights = [np.random.randn(i, j) for i, j in zip(self.layers[:-1], self.layers[1:])]

    def init_biases(self):
        """initialize biases randomly with normal distribution 1"""
        self.biases = [np.random.randn(1, i) for i in self.layers[1:]]

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

    def back_prop(self, X, Y, batch_size, eta=0.005, epochs=5, epoch_complete_call=None):
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
                self.update_weights(batchY, eta)

            if epoch_complete_call is not None:
                epoch_complete_call(epch)

    def update_weights(self, batchY, eta):
        """internal function"""
        delta = self.activations[-1] - batchY
        deltas = [delta]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * prime(self.Zs[i - 1])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] -= eta * np.dot(self.activations[i].T, deltas[i])
            self.biases[i] -= eta * np.mean(deltas[i], axis=0)
