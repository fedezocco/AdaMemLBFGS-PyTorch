"""
Implemented by Federico Zocco 
    Last update: 25/03/2020

References:
    [1] E. Stevens, L. Antiga, and T. Viehmann, "Deep learning with PyTorch",
        Manning Publications (2020).  
"""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.ReLU(out)
        out = self.layer2(out)
        return out
    

def MLPs_initializer(modelToBeInitialized, initializingModel):
    modelToBeInitialized.layer1.weight.data.copy_(initializingModel.layer1.weight.data)
    modelToBeInitialized.layer1.bias.data.copy_(initializingModel.layer1.bias.data)
    modelToBeInitialized.layer2.weight.data.copy_(initializingModel.layer2.weight.data)
    modelToBeInitialized.layer2.bias.data.copy_(initializingModel.layer2.bias.data)
    
    return modelToBeInitialized 
    
