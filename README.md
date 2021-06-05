# An Adaptive Memory Multi-Batch L-BFGS in PyTorch
The repository contains the source code of the paper "An adaptive memory multi-batch L-BFGS algorithm for neural network training" authored by Federico Zocco and Seán McLoone. The paper has been presented at the 21st IFAC World Congress, held in Berlin in July 12-17, 2020. 

If you use the code, please consider citing the corresponding paper: 
Federico Zocco, Seán McLoone,
An Adaptive Memory Multi-Batch L-BFGS Algorithm for Neural Network Training,
IFAC-PapersOnLine,
Volume 53, Issue 2,
2020,
Pages 8199-8204,
ISSN 2405-8963,
https://doi.org/10.1016/j.ifacol.2020.12.1996.  

Note that the "ETCH" case study, i.e. Experiment 2 of the paper, is not included in this repository because the "ETCH" dataset cannot be made publicly available for confidentiality reasons.


## Dependencies of folders and files 

The script "Algorithms_comparison.py" should be executed by the user to do the Monte Carlo simulations that compare the training algorithms on different datasets. It uses all the modules and data files inside the same folder of "Algorithms_comparison.py". It also uses the modules inside the folder "functions". 

The data files (i.e. the ".npy" files) are the datasets considered in Experiments 3, 4, and 6. The datasets for the remaining experiments are downloaded by the "Algorithms_comparison.py" script from the internet. 

The folder "Others" contains 2 scripts demonstrating how to do image rescaling and data augmentation. They are not used by the script "Algorithms_comparison.py". I have created and used this 2 scripts to rescale and augment the datasets as detailed in the paper. 




