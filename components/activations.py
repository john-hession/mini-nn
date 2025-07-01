import numpy as np

class ReLu:
    
    def __init__(self):
        self.output = None
    
    
    def forward(self, x):
        # max(x,0)
        self.x = x
        self.output = x * (x > 0)
        return self.output
    
    
    def backward(self, out_grad):
        # 1 if > 0, else 0
        
        if self.output is None:
            raise ValueError("Forward pass must be called before backward pass.")
        
        input_gradient = ((self.x > 0).astype(int)) * out_grad
        
        return input_gradient
        
    
class Sigmoid:
    
    def __init__(self):
        self.output = None
        
        
    def forward(self, x):
        # 1/(1+e^x)
        self.output = 1/(np.exp(-x) + 1)
        
        return self.output
    
    
    def backward(self, out_grad):
        # 1/(1+e^x)
        # wrt to x = (1+e^x)^-1 = -(1+e^x)^-1 + e^x = sigmoid(x) (1-sigmoid(x))
        if self.output is None:
            raise ValueError("Forward pass must be called before backward pass.")
        
        # multiply by output gradient to pass to next layer
        input_gradient = (self.output * (1-self.output)) * out_grad
        
        return input_gradient

class SoftMax:
    
    def __init__(self):
        self.output = None
        
    
    def forward(self, x):
        # e^x_i/ sum(e^x_j)
        self.x = x
        self.output = np.exp(x)/np.sum(np.exp(x))
        
        return self.output
    
    
    def backward(self, out_grad):
        # softmax derivative
        out_grad = np.reshape(out_grad, (1,-1))
        output = np.reshape(self.output, (1,-1))
        softmax_deriv = (output * np.identity(output.size) - output.transpose() @ output)
        
        return out_grad @ softmax_deriv
        
        
class Tanh:
    
    def __init__(self):
        self.output = None
    
    
    def forward(self, x):
        self.x = x
        self.output = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        
        return self.output
    
    
    def backward(self, out_grad):
        
        if self.output is None:
            raise ValueError("Forward pass must be called before backward pass.")
        
        # multiply by output gradient to pass to next layer
        input_gradient = (np.ones(self.output.shape) - np.pow(self.output,2)) * out_grad
        
        return input_gradient