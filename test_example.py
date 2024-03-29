import numpy as np

optimizer = str(input("Enter Optimizer: "))
if optimizer is None:
    print("Running test without Optimizer")
    from .basic_backprop import Network
    
elif optimizer == "RMSProp":
    print("Using RMSProp Optimizer")
    from .backprop_with_RMSProp import Network
    
elif optimizer == "Adagrad":
    print("Using Adagrad Optimizer")
    from .backprop_with_Adagrad import Network
    
elif optimizer == "Adam":
    print("Using Adam Optimizer")
    from .backprop_with_Adam import Network

elif optimizer == "Momentum":
    print("Using Backpropagation with Momentum")
    from .backprop_with_momentum import Network
    

def preeprocessing():
    X = np.random.random((5000, 2))     # 5000 total examples (4000 for training 1000 for testing)
    x1, x2 = X[:, 0], X[:, 1]
    x1x2 = x1 * x2
    y1 = np.sin(x1) * np.cos(x2)
    y2 = np.sin(x1x2)
    y3 = np.cos(x1x2)
    Y = np.array(list(zip(y1, y2, y3)))
    return X[:1000], Y[:1000], X[1000:], Y[1000:]


def check_accuracy(epoch):
    output = network.feed_forward(testX)
    # using root mean square error
    error = np.sqrt(np.mean(np.square(output - testY)))
    print(f"Epoch {epoch}: {error}")


if __name__ == '__main__':
    testX, testY, trainX, trainY = preeprocessing()
    network = Network([2, 32, 30, 3])
    network.back_prop(trainX, trainY, 5, epoch_complete_call=check_accuracy)
