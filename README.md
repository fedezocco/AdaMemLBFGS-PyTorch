# An Adaptive Memory Multi-Batch L-BFGS in PyTorch
The repository contains the source code of the paper "An adaptive memory multi-batch L-BFGS algorithm for neural network training" authored by Federico Zocco and Seán McLoone. The paper has been accepted at the 21st IFAC World Congress, which is going to be held in Berlin in July 12-17, 2020. 

I am going to upload the code as soon as the paper is published (roughly August/September 2020).

Link to the paper: 

Note that the "ETCH" case study, i.e. Experiment 2 of the paper, is not included in this repository because the "ETCH" dataset is not publishable.


## Dependencies of folders and files 

The script "Algorithms_comparison.py" should be executed by the user to do the Monte Carlo simulations comparing the training algorithms on different datasets. It uses all the modules and data file inside the same folder of "Algorithms_comparison.py". It also uses the modules inside the folder "functions". 

The data files (i.e. the ".npy" files) are the datasets considered in Experiments 3, 4, and 6. The datasets for the remaining experiments are downloaded by the "Algorithms_comparison.py" script from the internet. 

The folder "Others" contains 2 scripts demonstrating how to do image rescaling and data augmentation. They are not used by the script "Algorithms_comparison.py". I have created and used this 2 scripts to rescale and augment the datasets as detailed in the paper. 




