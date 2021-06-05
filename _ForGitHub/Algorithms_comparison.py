"""
Implemented by Federico Zocco 
    Last update: 25/03/2020

Source code of the paper: 
    Federico Zocco, Seán McLoone,
    An Adaptive Memory Multi-Batch L-BFGS Algorithm for Neural Network Training,
    IFAC-PapersOnLine,
    Volume 53, Issue 2,
    2020,
    Pages 8199-8204,
    ISSN 2405-8963,
    https://doi.org/10.1016/j.ifacol.2020.12.1996.
    (https://www.sciencedirect.com/science/article/pii/S2405896320326276)

This implementation is CUDA-compatible.

Requirements (first 2 are from the repository: https://github.com/hjmshi/PyTorch-LBFGS):
    (-) https://github.com/hjmshi/PyTorch-LBFGS/blob/master/examples/Neural_Networks/multi_batch_lbfgs_example.py 
    (-) https://github.com/hjmshi/PyTorch-LBFGS/tree/master/functions
    (-) Keras
    (-) NumPy
    (-) PyTorch

"""
# NOTE: "Adaptive" and "variable" are used as synonyms when referring to the L-BFGS memory. 

import sys
sys.path.append('../../functions/')
import time
import numpy as np
import torch
from sklearn import datasets
from keras.datasets import cifar10
from keras.datasets import mnist
import matplotlib.pyplot as plt
from multi_batch_lbfgs_Training import MultiBatchLBFGS_WithMemoryResetOption 


#%% Simulation settings and algorithm hyperparameters:

NumSimulations = 5
modelSelector = 'CNN' # either MLP or CNN

if (modelSelector == 'MLP'):   
    DatasetSelector = 'CANCER' # either IRIS or WAFERS or CANCER
    
    if(DatasetSelector == 'IRIS'):
        hidden_size = 15
        ### L-BFGS:
        NumIterations_LBFGS = 60
        overlap_ratio = 0.45 # must be in (0, 0.5)
        batch_size_LBFGS = 30
        lr_LBFGS = 1
        ghost_batch = 22
        # Adaptive memory L-BFGS:
        history_size_Init = 1 # called m_0 in the paper
        # L-BFGS:
        history_size = 10
        # Adam:
        NumIterations_Adam = 70
        batch_size_Adam = 20
        lr_Adam = 0.03
    if(DatasetSelector == 'WAFERS'):
        hidden_size = 5
        ### L-BFGS:
        NumIterations_LBFGS = 60
        overlap_ratio = 0.4 # must be in (0, 0.5)
        batch_size_LBFGS = 512
        lr_LBFGS = 1
        ghost_batch = 128
        # Adaptive memory L-BFGS:
        history_size_Init = 1 # called m_0 in the paper
        # L-BFGS:
        history_size = 10
        # Adam:
        NumIterations_Adam = 70
        batch_size_Adam = 64
        lr_Adam = 0.03
    if(DatasetSelector == 'CANCER'):
        hidden_size = 35
        ### L-BFGS:
        NumIterations_LBFGS = 200
        overlap_ratio = 0.45 # must be in (0, 0.5)
        batch_size_LBFGS = 256
        lr_LBFGS = 0.5
        ghost_batch = 80
        # Adaptive memory L-BFGS:
        history_size_Init = 1 # called m_0 in the paper
        # L-BFGS:
        history_size = 10
        # Adam:
        NumIterations_Adam = 200
        batch_size_Adam = 64
        lr_Adam = 0.02
        
        
if (modelSelector == 'CNN'): 
    DatasetSelector = 'CIFAR10' # either CIFAR10 or MNIST or TRASH
    
    if(DatasetSelector == 'CIFAR10'): 
        ### L-BFGS:
        NumIterations_LBFGS = 500
        overlap_ratio = 0.25 # must be in (0, 0.5)
        batch_size_LBFGS = 8192
        lr_LBFGS = 1
        ghost_batch = 128
        # Adaptive memory L-BFGS:
        history_size_Init = 1 # called m_0 in the paper
        # L-BFGS:
        history_size = 10
        # Adam:
        NumIterations_Adam = 1000
        batch_size_Adam = 128
        lr_Adam = 0.001
    if(DatasetSelector == 'MNIST'):        
        ### L-BFGS:
        NumIterations_LBFGS = 70
        overlap_ratio = 0.25 # must be in (0, 0.5)
        batch_size_LBFGS = 8192
        lr_LBFGS = 1
        ghost_batch = 128
        # Adaptive memory L-BFGS:
        history_size_Init = 1 # called m_0 in the paper
        # L-BFGS:
        history_size = 10
        # Adam:
        NumIterations_Adam = 80
        batch_size_Adam = 128
        lr_Adam = 0.001
    if(DatasetSelector == 'TRASH'):        
        ### L-BFGS:
        NumIterations_LBFGS = 900
        overlap_ratio = 0.25 # must be in (0, 0.5)
        batch_size_LBFGS = 8192
        lr_LBFGS = 1
        ghost_batch = 128
        # Adaptive memory L-BFGS:
        history_size_Init = 1 # called m_0 in the paper
        # L-BFGS:
        history_size = 10
        # Adam:
        NumIterations_Adam = 1200
        batch_size_Adam = 128
        lr_Adam = 0.001


# NOTE: the hyperparameters m_max, m_val, m_reset and alpha are 
# used as numerical values in "functions/LBFGS_WithMemoryResetOption.py/class LBFGS(Optimizer)"
#####################################################################   


#%% Load data and prepare test/training/validation sets:

if (modelSelector == 'MLP'): 
    if(DatasetSelector == 'IRIS'):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target 
        X = X.astype('float32') # float32 type required by Pytorch 
        y = y.astype('float32')
        # Standardization:
        X_mean = np.mean(X, axis=0)
        X_var = np.var(X, axis=0)
        X_standardized = (X - X_mean) / np.sqrt(X_var)
        # With m variable:
        # (training, test, validation) = (0.7, 0.15, 0.15)
        random_indices = np.random.permutation(range(X.shape[0]))
        train_indices_YESvarMem = random_indices[0:round(0.7*len(X))]
        X_train_YESvarMem = X[train_indices_YESvarMem]
        y_train_YESvarMem = y[train_indices_YESvarMem]
        # With m constant:
        # (training, test) = (0.85, 0.15)
        train_indices_NOvarMem = random_indices[0:round(0.85*len(X))]
        X_train_NOvarMem = X[train_indices_NOvarMem]
        y_train_NOvarMem = y[train_indices_NOvarMem]
        # With both constant and variable m:
        val_indices = random_indices[round(0.7*len(X)):round(0.85*len(X))]
        X_val = X[val_indices] # Passed as argument but not used whenever variable_memory='No' 
        y_val = y[val_indices] 
        test_indices = random_indices[round(0.85*len(X)):len(X)+1]
        X_test = X[test_indices]
        y_test = y[test_indices]
    if(DatasetSelector == 'WAFERS'):
        X_train = np.load('X_train_waferFaults.npy')
        X_test = np.load('X_test_waferFaults.npy')
        y_train = np.load('y_train_waferFaults.npy')
        y_test = np.load('y_test_waferFaults.npy')
        X_train = X_train.astype('float32') # float32 type required by Pytorch
        X_test = X_test.astype('float32')
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        # Standardization:
        X_train_mean = np.mean(X_train, axis=0)
        X_train_var = np.var(X_train, axis=0)
        X_train_standardized = (X_train - X_train_mean) / np.sqrt(X_train_var)
        X_test_mean = np.mean(X_test, axis=0)
        X_test_var = np.var(X_test, axis=0)
        X_test_standardized = (X_test - X_test_mean) / np.sqrt(X_test_var)
        # With m variable:
        # (take randomly len(X_test) samples for validation from X_train)
        train_indices = np.random.permutation(range(X_train.shape[0]))
        train_indices_YESvarMem = train_indices[0:(len(X_train)-len(X_test))]
        X_train_YESvarMem = X_train[train_indices_YESvarMem]
        y_train_YESvarMem = y_train[train_indices_YESvarMem]
        # With m constant:
        # (the whole X_train is for training)
        X_train_NOvarMem = X_train
        y_train_NOvarMem = y_train
        # With both constant and variable m:
        val_indices = train_indices[(len(X_train)-len(X_test)):len(X_train)+1]
        X_val = X_train[val_indices] # Passed as argument but not used whenever variable_memory='No' 
        y_val = y_train[val_indices]
    if(DatasetSelector == 'CANCER'):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X = X.astype('float32') # float32 type required by Pytorch 
        y = y.astype('float32')
        # Standardization:
        X_mean = np.mean(X, axis=0)
        X_var = np.var(X, axis=0)
        X_standardized = (X - X_mean) / np.sqrt(X_var)
        # With m variable:
        # (training, test, validation) = (0.7, 0.15, 0.15)
        random_indices = np.random.permutation(range(X.shape[0]))
        train_indices_YESvarMem = random_indices[0:round(0.7*len(X))]
        X_train_YESvarMem = X[train_indices_YESvarMem]
        y_train_YESvarMem = y[train_indices_YESvarMem]
        # With m constant:
        # (training, test) = (0.85, 0.15)
        train_indices_NOvarMem = random_indices[0:round(0.85*len(X))]
        X_train_NOvarMem = X[train_indices_NOvarMem]
        y_train_NOvarMem = y[train_indices_NOvarMem]
        # With both constant and variable m:
        val_indices = random_indices[round(0.7*len(X)):round(0.85*len(X))]
        X_val = X[val_indices] # Passed as argument but not used whenever variable_memory='No' 
        y_val = y[val_indices] 
        test_indices = random_indices[round(0.85*len(X)):len(X)+1]
        X_test = X[test_indices]
        y_test = y[test_indices]
        

if (modelSelector == 'CNN'):
    if(DatasetSelector == 'CIFAR10'):  
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = np.transpose(X_train, (0, 3, 1, 2)) # channel size must be the 2nd size in Pytorch
        X_test = np.transpose(X_test, (0, 3, 1, 2))
    if(DatasetSelector == 'MNIST'):
        ### Piece of code to resize images to be 32x32 for the CNN. Commented
        ### because done offline before running this code:
        #from rescale_images import rescale32x32
        (_, y_train), (_, y_test) = mnist.load_data()
        #X_train = np.reshape(X_train, (60000, 1, 28, 28))
        #X_test = np.reshape(X_test, (10000, 1, 28, 28))
        #X_train_mnist32x32 = rescale32x32(X_train) 
        #X_test_mnist32x32 = rescale32x32(X_test)           
        X_train = np.load('X_train_mnist32x32.npy')
        X_test = np.load('X_test_mnist32x32.npy')
    if(DatasetSelector == 'TRASH'):
        X = np.load('X_trash_augmented.npy')
        y = np.load('y_trash_augmented.npy')
        random_indices = np.random.permutation(range(X.shape[0]))
        random_indices_train = random_indices[0:round(0.85*len(X))]
        random_indices_test = random_indices[round(0.85*len(X)):len(X)+1]
        X_train = X[random_indices_train]
        X_test = X[random_indices_test]
        y_train = y[random_indices_train]
        y_test = y[random_indices_test]
    
    X_train = X_train.astype('float32') # float32 type required by Pytorch
    X_test = X_test.astype('float32')
    # Standardization:
    X_train = X_train/255
    X_test = X_test/255  
    # With m variable:
    # (take randomly len(X_test) samples for validation from X_train)
    train_indices = np.random.permutation(range(X_train.shape[0]))
    train_indices_YESvarMem = train_indices[0:(len(X_train)-len(X_test))]
    X_train_YESvarMem = X_train[train_indices_YESvarMem]
    y_train_YESvarMem = y_train[train_indices_YESvarMem]
    # With m constant:
    # (the whole X_train is for training)
    X_train_NOvarMem = X_train
    y_train_NOvarMem = y_train
    # With both constant and variable m:
    val_indices = train_indices[(len(X_train)-len(X_test)):len(X_train)+1]
    X_val = X_train[val_indices] # Passed as argument but not used whenever variable_memory='No' 
    y_val = y_train[val_indices]
    
    
#%% Check cuda availability
           
cuda = torch.cuda.is_available()

#%% Simulations:

for n_sim in range(NumSimulations): 
    
    X_train_YESvarMem = np.asarray(X_train_YESvarMem) 
    X_train_NOvarMem = np.asarray(X_train_NOvarMem)  
    y_train_YESvarMem = np.asarray(y_train_YESvarMem)
    y_train_NOvarMem = np.asarray(y_train_NOvarMem)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    # Create the models to train:
    if (modelSelector == 'MLP'):
        input_size = X_train_NOvarMem.shape[1] 
        num_classes = max(y_train_NOvarMem) + 1
        from Model_MLP import MLP
        if(cuda):
            torch.cuda.manual_seed(2018)
            model_forOptimizer1 = MLP(input_size, hidden_size, num_classes.astype('int')).cuda()
            model_forOptimizer2 = MLP(input_size, hidden_size, num_classes.astype('int')).cuda()
            model_forOptimizer3 = MLP(input_size, hidden_size, num_classes.astype('int')).cuda()
            model_forOptimizer4 = MLP(input_size, hidden_size, num_classes.astype('int')).cuda()
            model_forOptimizer5 = MLP(input_size, hidden_size, num_classes.astype('int')).cuda()
        else:
            torch.manual_seed(2018)
            model_forOptimizer1 = MLP(input_size, hidden_size, num_classes.astype('int'))
            model_forOptimizer2 = MLP(input_size, hidden_size, num_classes.astype('int'))
            model_forOptimizer3 = MLP(input_size, hidden_size, num_classes.astype('int'))
            model_forOptimizer4 = MLP(input_size, hidden_size, num_classes.astype('int'))
            model_forOptimizer5 = MLP(input_size, hidden_size, num_classes.astype('int'))
    
        # Initialize the MLPs with the same parameters:    
        from Model_MLP import MLPs_initializer
        if(cuda):
            torch.cuda.manual_seed(2018)
            sample_MLP = MLP(input_size, hidden_size, num_classes.astype('int')).cuda()
        else:
            torch.manual_seed(2018)
            sample_MLP = MLP(input_size, hidden_size, num_classes.astype('int'))
        model_forOptimizer1 = MLPs_initializer(model_forOptimizer1, sample_MLP) 
        model_forOptimizer2 = MLPs_initializer(model_forOptimizer2, sample_MLP)
        model_forOptimizer3 = MLPs_initializer(model_forOptimizer3, sample_MLP)
        model_forOptimizer4 = MLPs_initializer(model_forOptimizer4, sample_MLP)
        model_forOptimizer5 = MLPs_initializer(model_forOptimizer5, sample_MLP)
        
    if (modelSelector == 'CNN'):
        if(DatasetSelector == 'CIFAR10'): 
            NumChannels = 3
        if(DatasetSelector == 'MNIST'):
            NumChannels = 1
        if(DatasetSelector == 'TRASH'):
            NumChannels = 3
        
        from Model_ConvNet import ConvNet
        if(cuda):
            torch.cuda.manual_seed(2018)
            model_forOptimizer1 = ConvNet(NumChannels).cuda()
            model_forOptimizer2 = ConvNet(NumChannels).cuda()
            model_forOptimizer3 = ConvNet(NumChannels).cuda()
            model_forOptimizer4 = ConvNet(NumChannels).cuda()
            model_forOptimizer5 = ConvNet(NumChannels).cuda()
        else:
            torch.manual_seed(2018)
            model_forOptimizer1 = ConvNet(NumChannels)
            model_forOptimizer2 = ConvNet(NumChannels)
            model_forOptimizer3 = ConvNet(NumChannels)
            model_forOptimizer4 = ConvNet(NumChannels)
            model_forOptimizer5 = ConvNet(NumChannels)
    
        # Initialize the CNNs with the same parameters:    
        from Model_ConvNet import ConvNets_initializer
        if(cuda):
            torch.cuda.manual_seed(2018)
            sample_ConvNet = ConvNet(NumChannels).cuda()
        else:
            torch.manual_seed(2018)
            sample_ConvNet = ConvNet(NumChannels)
        model_forOptimizer1 = ConvNets_initializer(model_forOptimizer1, sample_ConvNet) 
        model_forOptimizer2 = ConvNets_initializer(model_forOptimizer2, sample_ConvNet)
        model_forOptimizer3 = ConvNets_initializer(model_forOptimizer3, sample_ConvNet)
        model_forOptimizer4 = ConvNets_initializer(model_forOptimizer4, sample_ConvNet)
        model_forOptimizer5 = ConvNets_initializer(model_forOptimizer5, sample_ConvNet)

    
    
    # Optimization step:
    
    # (1) Multi batch L-BFGS without memory resetting and m constant:
    clockStart_MultiBatchNOreset = time.clock() 
    v_trainLoss_MultiBatchNOreset, _, v_testLoss_MultiBatchNOreset, v_testAccuracy_MultiBatchNOreset, _, v_memory_MultiBatchNOreset, v_numPairs_MultiBatchNOreset = MultiBatchLBFGS_WithMemoryResetOption(cuda, model_forOptimizer1, X_train_NOvarMem, y_train_NOvarMem, X_val, y_val, X_test, y_test, history_size, NumIterations_LBFGS, ghost_batch, batch_size_LBFGS, overlap_ratio, lr_LBFGS, history_reset='No', variable_memory='No')
    clockEnd_MultiBatchNOreset = time.clock() - clockStart_MultiBatchNOreset 
    
    # (2) Multi batch L-BFGS without memory resetting and Variable Memory:
    clockStart_MultiBatchNOresetYESvarMem = time.clock() 
    v_trainLoss_MultiBatchNOresetYESvarMem, v_valLoss_MultiBatchNOresetYESvarMem, v_testLoss_MultiBatchNOresetYESvarMem, v_testAccuracy_MultiBatchNOresetYESvarMem, _, v_memory_MultiBatchNOresetYESvarMem, v_numPairs_MultiBatchNOresetYESvarMem = MultiBatchLBFGS_WithMemoryResetOption(cuda, model_forOptimizer2, X_train_YESvarMem, y_train_YESvarMem, X_val, y_val, X_test, y_test, history_size_Init, NumIterations_LBFGS, ghost_batch, batch_size_LBFGS, overlap_ratio, lr_LBFGS, history_reset='No', variable_memory='Yes')
    clockEnd_MultiBatchNOresetYESvarMem = time.clock() - clockStart_MultiBatchNOresetYESvarMem
    
    # (3) Multi batch L-BFGS with memory resetting and m constant:
    clockStart_MultiBatchYESreset = time.clock() 
    v_trainLoss_MultiBatchYESreset, _, v_testLoss_MultiBatchYESreset, v_testAccuracy_MultiBatchYESreset, _, v_memory_MultiBatchYESreset, v_numPairs_MultiBatchYESreset = MultiBatchLBFGS_WithMemoryResetOption(cuda, model_forOptimizer3, X_train_NOvarMem, y_train_NOvarMem, X_val, y_val, X_test, y_test, history_size, NumIterations_LBFGS, ghost_batch, batch_size_LBFGS, overlap_ratio, lr_LBFGS, history_reset='Yes', variable_memory='No')
    clockEnd_MultiBatchYESreset = time.clock() - clockStart_MultiBatchYESreset
    
    # (4) Multi batch L-BFGS with memory resetting and Variable Memory:
    clockStart_MultiBatchYESresetYESvarMem = time.clock() 
    v_trainLoss_MultiBatchYESresetYESvarMem, v_valLoss_MultiBatchYESresetYESvarMem, v_testLoss_MultiBatchYESresetYESvarMem, v_testAccuracy_MultiBatchYESresetYESvarMem, _, v_memory_MultiBatchYESresetYESvarMem, v_numPairs_MultiBatchYESresetYESvarMem = MultiBatchLBFGS_WithMemoryResetOption(cuda, model_forOptimizer4, X_train_YESvarMem, y_train_YESvarMem, X_val, y_val, X_test, y_test, history_size_Init, NumIterations_LBFGS, ghost_batch, batch_size_LBFGS, overlap_ratio, lr_LBFGS, history_reset='Yes', variable_memory='Yes')
    clockEnd_MultiBatchYESresetYESvarMem = time.clock() - clockStart_MultiBatchYESresetYESvarMem
    
    # (5) Adam:
    from Adam_Training import adam_TrainingLoop 
    X_train_NOvarMem = torch.tensor(X_train_NOvarMem) # torch.optim.Adam expects Tensors as arguments 
    X_test = torch.tensor(X_test)
    y_train_NOvarMem = torch.tensor(y_train_NOvarMem).long().squeeze() # label vector expected to be long()
    y_test = torch.tensor(y_test).long().squeeze()
    clockStart_Adam = time.clock()
    v_trainLoss_Adam, v_testLoss_Adam, v_testAccuracy_Adam, NumEpochs_Adam = adam_TrainingLoop(cuda, NumIterations_Adam, model_forOptimizer5, lr_Adam, batch_size_Adam, X_train_NOvarMem, y_train_NOvarMem, X_test, y_test)
    clockEnd_Adam = time.clock() - clockStart_Adam
    
    # Store results from each simulation:
    if n_sim == 0: 
        # (1):
        M_trainLoss_MultiBatchNOreset = np.asarray([v_trainLoss_MultiBatchNOreset])
        M_testLoss_MultiBatchNOreset = np.asarray([v_testLoss_MultiBatchNOreset])
        M_testAccuracy_MultiBatchNOreset = np.asarray([v_testAccuracy_MultiBatchNOreset])
        M_memory_MultiBatchNOreset = np.asarray([v_memory_MultiBatchNOreset])
        v_Time_MultiBatchNOreset = clockEnd_MultiBatchNOreset
        # (2):
        M_trainLoss_MultiBatchNOresetYESvarMem = np.asarray([v_trainLoss_MultiBatchNOresetYESvarMem])
        M_valLoss_MultiBatchNOresetYESvarMem = np.asarray([v_valLoss_MultiBatchNOresetYESvarMem])
        M_testLoss_MultiBatchNOresetYESvarMem = np.asarray([v_testLoss_MultiBatchNOresetYESvarMem])
        M_testAccuracy_MultiBatchNOresetYESvarMem = np.asarray([v_testAccuracy_MultiBatchNOresetYESvarMem])
        M_memory_MultiBatchNOresetYESvarMem = np.asarray([v_memory_MultiBatchNOresetYESvarMem])
        v_Time_MultiBatchNOresetYESvarMem = clockEnd_MultiBatchNOresetYESvarMem
        # (3):
        M_trainLoss_MultiBatchYESreset = np.asarray([v_trainLoss_MultiBatchYESreset])
        M_testLoss_MultiBatchYESreset = np.asarray([v_testLoss_MultiBatchYESreset])
        M_testAccuracy_MultiBatchYESreset = np.asarray([v_testAccuracy_MultiBatchYESreset])
        M_memory_MultiBatchYESreset = np.asarray([v_memory_MultiBatchYESreset])
        v_Time_MultiBatchYESreset = clockEnd_MultiBatchYESreset
        # (4):
        M_trainLoss_MultiBatchYESresetYESvarMem = np.asarray([v_trainLoss_MultiBatchYESresetYESvarMem])
        M_valLoss_MultiBatchYESresetYESvarMem = np.asarray([v_valLoss_MultiBatchYESresetYESvarMem])
        M_testLoss_MultiBatchYESresetYESvarMem = np.asarray([v_testLoss_MultiBatchYESresetYESvarMem])
        M_testAccuracy_MultiBatchYESresetYESvarMem = np.asarray([v_testAccuracy_MultiBatchYESresetYESvarMem])
        M_memory_MultiBatchYESresetYESvarMem = np.asarray([v_memory_MultiBatchYESresetYESvarMem])
        v_Time_MultiBatchYESresetYESvarMem = clockEnd_MultiBatchYESresetYESvarMem
        # (5):
        M_trainLoss_Adam = np.asarray([v_trainLoss_Adam])
        M_testLoss_Adam = np.asarray([v_testLoss_Adam])
        M_testAccuracy_Adam = np.asarray([v_testAccuracy_Adam])
        v_Time_Adam = clockEnd_Adam
        
        
    else:
        # (1):
        M_trainLoss_MultiBatchNOreset = np.append(M_trainLoss_MultiBatchNOreset, np.asarray([v_trainLoss_MultiBatchNOreset]), axis=0)
        M_testLoss_MultiBatchNOreset = np.append(M_testLoss_MultiBatchNOreset, np.asarray([v_testLoss_MultiBatchNOreset]), axis=0)
        M_testAccuracy_MultiBatchNOreset = np.append(M_testAccuracy_MultiBatchNOreset, np.asarray([v_testAccuracy_MultiBatchNOreset]), axis=0)
        M_memory_MultiBatchNOreset = np.append(M_memory_MultiBatchNOreset, np.asarray([v_memory_MultiBatchNOreset]), axis=0)
        v_Time_MultiBatchNOreset = np.append(v_Time_MultiBatchNOreset, clockEnd_MultiBatchNOreset)
        # (2):
        M_trainLoss_MultiBatchNOresetYESvarMem = np.append(M_trainLoss_MultiBatchNOresetYESvarMem, np.asarray([v_trainLoss_MultiBatchNOresetYESvarMem]), axis=0)
        M_valLoss_MultiBatchNOresetYESvarMem = np.append(M_valLoss_MultiBatchNOresetYESvarMem, np.asarray([v_valLoss_MultiBatchNOresetYESvarMem]), axis=0)
        M_testLoss_MultiBatchNOresetYESvarMem = np.append(M_testLoss_MultiBatchNOresetYESvarMem, np.asarray([v_testLoss_MultiBatchNOresetYESvarMem]), axis=0)
        M_testAccuracy_MultiBatchNOresetYESvarMem = np.append(M_testAccuracy_MultiBatchNOresetYESvarMem, np.asarray([v_testAccuracy_MultiBatchNOresetYESvarMem]), axis=0)
        M_memory_MultiBatchNOresetYESvarMem = np.append(M_memory_MultiBatchNOresetYESvarMem, np.asarray([v_memory_MultiBatchNOresetYESvarMem]), axis=0)
        v_Time_MultiBatchNOresetYESvarMem = np.append(v_Time_MultiBatchNOresetYESvarMem, clockEnd_MultiBatchNOresetYESvarMem)
        # (3):
        M_trainLoss_MultiBatchYESreset = np.append(M_trainLoss_MultiBatchYESreset, np.asarray([v_trainLoss_MultiBatchYESreset]), axis=0)
        M_testLoss_MultiBatchYESreset = np.append(M_testLoss_MultiBatchYESreset, np.asarray([v_testLoss_MultiBatchYESreset]), axis=0)
        M_testAccuracy_MultiBatchYESreset = np.append(M_testAccuracy_MultiBatchYESreset, np.asarray([v_testAccuracy_MultiBatchYESreset]), axis=0)
        M_memory_MultiBatchYESreset = np.append(M_memory_MultiBatchYESreset, np.asarray([v_memory_MultiBatchYESreset]), axis=0)
        v_Time_MultiBatchYESreset = np.append(v_Time_MultiBatchYESreset, clockEnd_MultiBatchYESreset)
        # (4):
        M_trainLoss_MultiBatchYESresetYESvarMem = np.append(M_trainLoss_MultiBatchYESresetYESvarMem, np.asarray([v_trainLoss_MultiBatchYESresetYESvarMem]), axis=0)
        M_valLoss_MultiBatchYESresetYESvarMem = np.append(M_valLoss_MultiBatchYESresetYESvarMem, np.asarray([v_valLoss_MultiBatchYESresetYESvarMem]), axis=0)
        M_testLoss_MultiBatchYESresetYESvarMem = np.append(M_testLoss_MultiBatchYESresetYESvarMem, np.asarray([v_testLoss_MultiBatchYESresetYESvarMem]), axis=0)
        M_testAccuracy_MultiBatchYESresetYESvarMem = np.append(M_testAccuracy_MultiBatchYESresetYESvarMem, np.asarray([v_testAccuracy_MultiBatchYESresetYESvarMem]), axis=0)
        M_memory_MultiBatchYESresetYESvarMem = np.append(M_memory_MultiBatchYESresetYESvarMem, np.asarray([v_memory_MultiBatchYESresetYESvarMem]), axis=0)
        v_Time_MultiBatchYESresetYESvarMem = np.append(v_Time_MultiBatchYESresetYESvarMem, clockEnd_MultiBatchYESresetYESvarMem)
        # (5):
        M_trainLoss_Adam = np.append(M_trainLoss_Adam, np.asarray([v_trainLoss_Adam]), axis=0)
        M_testLoss_Adam = np.append(M_testLoss_Adam, np.asarray([v_testLoss_Adam]), axis=0)
        M_testAccuracy_Adam = np.append(M_testAccuracy_Adam, np.asarray([v_testAccuracy_Adam]), axis=0)
        v_Time_Adam = np.append(v_Time_Adam, clockEnd_Adam)  



#%% Plots:
  
plt.rc('font', size=25)

# (1) training loss:
fig, ax = plt.subplots()
ax.plot(np.arange(NumIterations_LBFGS)+1, np.mean(M_trainLoss_MultiBatchNOreset, axis=0), 'b-.', markersize=12, linewidth=4, label='MB')
ax.plot(np.arange(NumIterations_LBFGS)+1, np.mean(M_trainLoss_MultiBatchNOresetYESvarMem, axis=0), 'r-.', markersize=12, linewidth=4, label='MB-AM')
ax.plot(np.arange(NumIterations_LBFGS)+1, np.mean(M_trainLoss_MultiBatchYESreset, axis=0), 'g--', markersize=12, linewidth=4, label='MB-R')
ax.plot(np.arange(NumIterations_LBFGS)+1, np.mean(M_trainLoss_MultiBatchYESresetYESvarMem, axis=0), 'k-.', markersize=12, linewidth=4, label='MB-AMR')
ax.plot(np.arange(NumIterations_Adam)+1, np.mean(M_trainLoss_Adam, axis=0), 'm-.', markersize=12, linewidth=4, label='Adam')
ax.set(xlabel='Iteration', ylabel='Training loss')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=4, prop={'size': 11})
ax.set_ylim([0,4])
ax.set_xlim([0,max(NumIterations_LBFGS, NumIterations_Adam)])

# (2) test loss:
fig, ax = plt.subplots()
ax.plot(np.arange(NumIterations_LBFGS)+1, np.mean(M_testLoss_MultiBatchNOreset, axis=0), 'b--', markersize=12, linewidth=4, label='MB')
ax.plot(np.arange(NumIterations_LBFGS)+1, np.mean(M_testLoss_MultiBatchNOresetYESvarMem, axis=0), 'r--', markersize=12, linewidth=4, label='MB-AM')
ax.plot(np.arange(NumIterations_LBFGS)+1, np.mean(M_testLoss_MultiBatchYESreset, axis=0), 'g--', markersize=12, linewidth=4, label='MB-R')
ax.plot(np.arange(NumIterations_LBFGS)+1, np.mean(M_testLoss_MultiBatchYESresetYESvarMem, axis=0), 'k--', markersize=12, linewidth=4, label='MB-AMR')
ax.plot(np.arange(NumIterations_Adam)+1, np.mean(M_testLoss_Adam, axis=0), 'm--', markersize=12, linewidth=4, label='Adam')
ax.set(xlabel='Iteration', ylabel='Test loss')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=4, prop={'size': 11})
ax.set_ylim([0,4])
ax.set_xlim([0,max(NumIterations_LBFGS, NumIterations_Adam)])

# (3) test accuracy:
fig, ax = plt.subplots()
ax.plot(np.arange(NumIterations_LBFGS), np.mean(M_testAccuracy_MultiBatchNOreset, axis=0), 'b--', markersize=12, linewidth=4, label='MB')
ax.plot(np.arange(NumIterations_LBFGS), np.mean(M_testAccuracy_MultiBatchNOresetYESvarMem, axis=0), 'r--', markersize=12, linewidth=4, label='MB-AM')
ax.plot(np.arange(NumIterations_LBFGS), np.mean(M_testAccuracy_MultiBatchYESreset, axis=0), 'g--', markersize=12, linewidth=4, label='MB-R')
ax.plot(np.arange(NumIterations_LBFGS), np.mean(M_testAccuracy_MultiBatchYESresetYESvarMem, axis=0), 'k--', markersize=12, linewidth=4, label='MB-AMR')
ax.plot(np.arange(NumIterations_Adam), np.mean(M_testAccuracy_Adam, axis=0), 'm--', markersize=12, linewidth=4, label='Adam')
ax.set(xlabel='Iteration', ylabel='Test accuracy (%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=4, prop={'size': 11})
ax.set_xlim([0,max(NumIterations_LBFGS, NumIterations_Adam)]) 

# (4) m_k:
fig, ax = plt.subplots()
ax.plot(np.arange(NumIterations_LBFGS)+1, v_memory_MultiBatchNOreset, 'b--', markersize=12, linewidth=4, label='MB')
ax.plot(np.arange(NumIterations_LBFGS)+1, v_memory_MultiBatchNOresetYESvarMem, 'r--', markersize=12, linewidth=4, label='MB-AM')
ax.plot(np.arange(NumIterations_LBFGS)+1, v_memory_MultiBatchYESreset, 'g--', markersize=12, linewidth=4, label='MB-R')
ax.plot(np.arange(NumIterations_LBFGS)+1, v_memory_MultiBatchYESresetYESvarMem, 'k--', markersize=12, linewidth=4, label='MB-AMR')
ax.set(xlabel='Iteration', ylabel='Memory size')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=4, prop={'size': 11})
ax.set_xlim([0,max(NumIterations_LBFGS, NumIterations_Adam)])

# (5) q_k:
fig, ax = plt.subplots()
ax.plot(np.arange(NumIterations_LBFGS)+1, v_numPairs_MultiBatchNOreset, 'b--', markersize=12, linewidth=4, label='MB')
ax.plot(np.arange(NumIterations_LBFGS)+1, v_numPairs_MultiBatchNOresetYESvarMem, 'r--', markersize=12, linewidth=4, label='MB-AM')
ax.plot(np.arange(NumIterations_LBFGS)+1, v_numPairs_MultiBatchYESreset, 'g--', markersize=12, linewidth=4, label='MB-R')
ax.plot(np.arange(NumIterations_LBFGS)+1, v_numPairs_MultiBatchYESresetYESvarMem, 'k--', markersize=12, linewidth=4, label='MB-AMR')
ax.set(xlabel='Iteration', ylabel='Curvature pairs')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=4, prop={'size': 11})
ax.set_xlim([0,max(NumIterations_LBFGS, NumIterations_Adam)])
