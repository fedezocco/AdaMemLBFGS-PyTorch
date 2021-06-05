"""
Implemented by Federico Zocco 
    Last update: 25/03/2020

Example on how to rescale the images of a dataset to be 32x32. 
"""

import numpy as np
import cv2

def rescale32x32(X_toRescale): # expected input with format (n_samples, channels, imageSize1, imageSize2)
    
    for sample in range(len(X_toRescale)):
        sample_toRescale = X_toRescale[sample]
        sample_toRescale = np.transpose(sample_toRescale, (1, 2, 0))
        sample_rescaled = cv2.resize(sample_toRescale, dsize=(32, 32))
        if (len(X_toRescale[0,:,0,0]) == 1):
            sample_rescaled = np.reshape(sample_rescaled, (1, 32, 32))
        if (len(X_toRescale[0,:,0,0]) == 3):
            sample_rescaled = np.transpose(sample_rescaled, (2, 0, 1))
            
        if sample == 0:
            X_rescaled = np.asarray([sample_rescaled])
        else:
            X_rescaled = np.append(X_rescaled, np.asarray([sample_rescaled]), axis=0)
                
    return X_rescaled 

