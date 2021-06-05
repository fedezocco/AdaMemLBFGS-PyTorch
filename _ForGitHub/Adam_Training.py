"""
Implemented by Federico Zocco 
    Last update: 25/03/2020
    
References:
    [1] D. Kingma and J. Ba, "Adam: A method for stochastic 
        optimization", arXiv preprint: arXiv:1412.6980 (2014).
    [2] R. Sebastian, "An overview of gradient descent optimization algorithms", 
        arXiv preprint: arXiv:1609.04747 (2016).
    [3] E. Stevens, L. Antiga, and T. Viehmann, "Deep learning with PyTorch",
        Manning Publications (2020).    
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math


def adam_TrainingLoop(cuda, NumIterations, model, lr, batch_size, X_train, y_train, X_test, y_test):
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    # Cost function
    loss_fn = nn.CrossEntropyLoss()
    
    v_trainLoss = []
    v_testLoss = []
    v_testAccuracy = []
    random_indices = np.random.permutation(range(X_train.shape[0]))  
    
    for n_iter in range(NumIterations):
        
        model.train()
        # training set loss:    
        if random_indices.shape[0] < 1: # i.e. an epoch is completed
            random_indices = np.random.permutation(range(X_train.shape[0]))
            
        batch_indices = random_indices[0:batch_size]
        random_indices = np.delete(random_indices, np.array(range(batch_size)))
        # (a) forward step
        if(cuda):
            outputs_trainBatch = model(X_train[batch_indices].cuda()) 
        else:
            outputs_trainBatch = model(X_train[batch_indices]) 
        # (b) evaluate the loss
        if(cuda):
            loss_trainBatch = loss_fn(outputs_trainBatch, y_train[batch_indices].cuda())
        else:
            loss_trainBatch = loss_fn(outputs_trainBatch, y_train[batch_indices])
        # (c) reset gradients
        optimizer.zero_grad()
        # (d) backward
        loss_trainBatch.backward()
        # (e) update the weights
        optimizer.step()
        v_trainLoss = np.append(v_trainLoss, loss_trainBatch.item())
        
        # test set loss:
        if(cuda):
            outputs_test = model(X_test.cuda()) 
        else:
            outputs_test = model(X_test)
        if(cuda):
            loss_test = loss_fn(outputs_test, y_test.cuda())
        else:
            loss_test = loss_fn(outputs_test, y_test)
        v_testLoss = np.append(v_testLoss, loss_test.item())
        
        # test set accuracy:
        model.eval()
        if(cuda):
            testAccuracy = np.mean(np.asarray(np.equal(torch.argmax(outputs_test, dim=1).cpu(), y_test.cpu())), axis=0)*100
        else:
           testAccuracy = np.mean(np.asarray(np.equal(torch.argmax(outputs_test, dim=1), y_test)), axis=0)*100 
        v_testAccuracy = np.append(v_testAccuracy, testAccuracy)
    
        NumEpochs = NumIterations/(math.ceil(X_train.shape[0]/batch_size))
        print('Adam Iter:', n_iter+1, 'Training loss:', loss_trainBatch.item(), 
          'Test loss:', loss_test.item(), 'Test accuracy:', testAccuracy)
    
    return v_trainLoss, v_testLoss, v_testAccuracy, NumEpochs 
        
          