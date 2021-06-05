"""
Implemented by Federico Zocco 
    Last update: 25/03/2020

References:
    [1] E. Stevens, L. Antiga, and T. Viehmann, "Deep learning with PyTorch",
        Manning Publications (2020).  
"""

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, NumChannels):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(NumChannels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    

def ConvNets_initializer(modelToBeInitialized, initializingModel):
    modelToBeInitialized.conv1.weight.data.copy_(initializingModel.conv1.weight.data)
    modelToBeInitialized.conv1.bias.data.copy_(initializingModel.conv1.bias.data)
    modelToBeInitialized.conv2.weight.data.copy_(initializingModel.conv2.weight.data)
    modelToBeInitialized.conv2.bias.data.copy_(initializingModel.conv2.bias.data)
    modelToBeInitialized.fc1.weight.data.copy_(initializingModel.fc1.weight.data)
    modelToBeInitialized.fc1.bias.data.copy_(initializingModel.fc1.bias.data)
    modelToBeInitialized.fc2.weight.data.copy_(initializingModel.fc2.weight.data)
    modelToBeInitialized.fc2.bias.data.copy_(initializingModel.fc2.bias.data)
    
    return modelToBeInitialized