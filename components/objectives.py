import numpy as np

class MSELoss:
    
    def __init__(self, observed, predicted):
        
        self.observed = observed
        self.predicted = predicted
        self.error = None
        
        
    def forward(self):
        
        # 1/n sum(Y-Y_hat)^2
        self.error = np.mean(np.square(self.observed - self.predicted))
        
        return self.error
    
    
    def backward(self, learning_rate):
        
        # derivative wrt to y(input) = 2/n(Y-Y_hat) 
        # derivative wrt to y_hat(true values) is irrelevant
        input_derivative = (2/self.observed.shape[0])*(self.observed - self.predicted)
        
        return input_derivative * learning_rate
    

class CrossEntropyLoss:
    
    def __init__(self, observed, predicted):
        
        self.observed = observed
        self.predicted = predicted
        self.error = None
    
    ## TODO: Cross entropy, accuracy