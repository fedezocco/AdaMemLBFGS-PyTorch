"""
Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere

Copyright (c) 2018 Hao-Jun Michael Shi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.    

Modified by Federico Zocco 
    Last update: 25/03/2020
    
    References:
        [1] A. Berahas, J. Nocedal, and M. Takác, "A multi-batch L-BFGS method for 
            machine learning", in "Advances in Neural Information Processing Systems", 
            1055-1063 (2016).
        [2] S. McLoone and G. Irwin, "A variable memory quasi-Newton training 
            algorithm", Neural Processing Letters, 9(1), 77-89 (1999).
        [3] A. Wilson, R. Roelofs, M. Stern, N. Srebro, and B. Recht, "The 
            marginal value of adaptive gradient methods in machine learning", in 
            "Advances in Neural Information Processing Systems", 4148-4158 (2017).
        [4] R. Bollapragada, D. Mudigere, J. Nocedal, H.M. Shi, and P.T.P. Tang,
            "A progressive batching L-BFGS method for machine learning", arXiv 
            preprint: arXiv:1802.05374 (2018).
"""

import numpy as np
import torch.optim

from utils import compute_stats, get_grad
from LBFGS_WithMemoryResetOption import LBFGS


def MultiBatchLBFGS_WithMemoryResetOption(cuda, model, X_train, y_train, X_val, y_val, X_test, y_test, history_size=10, max_iter=50, ghost_batch=2, batch_size=1, overlap_ratio=0.25, lr=1, history_reset='No', variable_memory='No'):
        
    #%% Define helper functions
    
    # Forward pass
    if(cuda):
        opfun = lambda X: model.forward(torch.from_numpy(X).cuda())
    else:
        opfun = lambda X: model.forward(torch.from_numpy(X))
    
    # Forward pass through the network given the input
    if(cuda):
        predsfun = lambda op: np.argmax(op.cpu().data.numpy(), 1)
    else:
        predsfun = lambda op: np.argmax(op.data.numpy(), 1)
    
    # Do the forward pass, then compute the accuracy
    accfun   = lambda op, y: np.mean(np.equal(predsfun(op), y.squeeze()))*100
    
    #%% Define optimizer
    
    optimizer = LBFGS(model.parameters(), lr, history_size, line_search='None', debug=True)
    
    #%% Main training loop
    
    Ok_size = int(overlap_ratio*batch_size)
    Nk_size = int((1 - 2*overlap_ratio)*batch_size)
    
    # sample previous overlap gradient
    random_index = np.random.permutation(range(X_train.shape[0]))
    Ok_prev = random_index[0:Ok_size]
    g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X_train[Ok_prev], y_train[Ok_prev], opfun)
    
    # main loop
    v_trainLoss = []
    v_valLoss = []
    v_testLoss = []
    v_testAccuracy = []
    v_history_size = []
    v_numPairs = []
    
    for n_iter in range(max_iter):
        
        # training mode
        model.train()
        
        # sample current non-overlap and next overlap gradient
        random_index = np.random.permutation(range(X_train.shape[0]))
        Ok = random_index[0:Ok_size]
        Nk = random_index[Ok_size:(Ok_size + Nk_size)]
        
        # compute overlap gradient and objective
        g_Ok, obj_Ok = get_grad(optimizer, X_train[Ok], y_train[Ok], opfun)
        
        # compute non-overlap gradient and objective
        g_Nk, obj_Nk = get_grad(optimizer, X_train[Nk], y_train[Nk], opfun)
        
        # compute accumulated gradient over sample
        g_Sk = overlap_ratio*(g_Ok_prev + g_Ok) + (1 - 2*overlap_ratio)*g_Nk
            
        # two-loop recursion to compute search direction
        p = optimizer.two_loop_recursion(-g_Sk)
                    
        # perform line search step
        lr = optimizer.step(p, g_Ok, g_Sk=g_Sk)
        
        # compute previous overlap gradient for next sample
        Ok_prev = Ok
        g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X_train[Ok_prev], y_train[Ok_prev], opfun)
        
        # curvature update
        state, history_size_new, n_dirs = optimizer.curvature_update(g_Ok_prev, eps=0.2, damping=True, history_reset=history_reset, v_valLoss=v_valLoss)
        
        # compute statistics
        model.eval()
        train_loss, val_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_val, y_val, X_test, 
                                                        y_test, opfun, accfun, ghost_batch)
                
        v_trainLoss.append(train_loss) 
        v_testLoss.append(test_loss)
        v_testAccuracy.append(test_acc)
        v_history_size.append(history_size_new)
        v_numPairs.append(n_dirs)
        if variable_memory == 'Yes':
            v_valLoss.append(val_loss)
        
        print('History reset:', history_reset, 'Variable memory:', variable_memory, 'Iter:',n_iter+1, 'Training loss:', train_loss, 
          'Test loss:', test_loss, 'Test accuracy:', test_acc, 'lr:', lr)
        
    
    return v_trainLoss, v_valLoss, v_testLoss, v_testAccuracy, state, v_history_size, v_numPairs
