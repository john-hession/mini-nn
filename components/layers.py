
import numpy as np
#https://medium.com/@z.shaked/building-micrograd-with-vectorization-a-step-by-step-guide-dcf897fc54cb
class Linear:
    def __init__(self, in_size, out_size):
        self.weights = np.random.normal(0, np.sqrt(2/in_size), (in_size, out_size))
        self.biases = np.zeros((1,out_size))
        self.x = None
        self.output = None
        
    def forward(self, x):
        ## x w^t + b
        self.x = x
        self.output = np.matmul(x,self.weights) + self.biases
        
        return self.output
    
    def backward(self, out_grad, learning_rate):
        ## x w^t + b, derivatives trivial bx terms drop
        # derivative wrt to x = w^t
        # derivative wrt to w = x
        # derivative wrt to b = 1
        
        if self.output is None:
            raise ValueError("Forward pass must be called before backward pass.")
        
        # Calculate gradients
       
        weights_gradient = np.dot(self.x.T, out_grad)
        bias_gradient = np.sum(out_grad, axis=0, keepdims=True)
        input_gradient = np.dot(out_grad, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * bias_gradient
        
        return input_gradient
    

## TODO: Conv Layers, Attention, RWKV, AFT, Flash Attention?