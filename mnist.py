import random
import numpy as np
import mnist_loader

# This program is a python implementation of a Convolutional Neural Network.
# It performs digit recognition, MNIST dataset was used to train the model.

class CNN:

    # initialises the neural network 
    def __init__ (self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]

    # given input data it predicts the output
    def feedForward(self,a):
        for w,b in zip(self.weights,self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,alpha,test_data=None):
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, alpha):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(alpha/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(alpha/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
            

    # evaluate the accuracy of the model
    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedForward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    # Viewing the weights and biases in the network.
    def printAll(self):
        print(self.weights)
        print(self.biases)
    
    
# sigmoid function used for logistic regression.  
def sigmoid(g) :
    return (1.0/(1.0 + np.exp(-g)))

# derivative of the sigmoid function used in updating weights and biases.
def sigmoid_prime(g):
    return sigmoid(g)*(1-sigmoid(g))

if __name__ == '__main__':
    training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(validation_data)
    net = CNN([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
