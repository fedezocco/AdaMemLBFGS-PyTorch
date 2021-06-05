"""
Implemented by Federico Zocco 
    Last update: 25/03/2020

Example on how to do data augmentation using Keras. Here the "TRASH" dataset is 
considered. The final dataset is 21 times bigger than the original one 
(i.e. 21*2527 samples/images).
"""

import random
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

X_trash_augmented = X_trash
y_trash_augmented = y_trash
for i in range(20*len(X_trash)):
    sample = random.randrange(len(X_trash))
    new_image = train_datagen.random_transform(X_trash[sample,:,:,:])
    X_trash_augmented = np.append(X_trash_augmented, np.asarray([new_image]), axis=0)
    y_trash_augmented = np.append(y_trash_augmented, np.asarray([y_trash[sample]]), axis=0) 