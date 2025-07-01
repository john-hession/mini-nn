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
        
        return input_derivative
    

class CrossEntropyLoss:
    
    def __init__(self, observed, predicted):
        
        self.observed = observed
        self.predicted = predicted
        self.error = None
        
    def forward(self):
        # - sum(observed * log(predicted))
        return np.sum(self.observed * np.log(self.predicted + 0.00001)) * -1
    
    def backward(self):
        # derivative wrt to q: 
        # - sum(observed * log(predicted)) only care about 1 class
        # -observed_i * log(predicted_i) d/pred_i
        # - observed_i * 1/predicted_i
        
        input_derivative =  self.predicted - self.observed 
        return input_derivative