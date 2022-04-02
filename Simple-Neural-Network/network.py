import math

def train(network, x_train, y_train, loss, loss_prime, epoches=1000, learning_rate=0.1, verbose=True):
    error = 0

    for e in range(epoches):
        
        for x, y in zip(x_train, y_train):
            # Forward
            output = x
            for layer in network:
                output = layer.forward(output)


            error += loss(y, output)


            # Backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
   
        if verbose: 
            error /= len(x_train)
            print("{}, error = {}".format(math.trunc(e+1/epoches), error))

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)     
    return output     
