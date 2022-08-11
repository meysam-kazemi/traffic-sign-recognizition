# =============================================================================
# Importing libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import tensorflow as tf
import PIL

# =============================================================================
# Loading data
# =============================================================================
path="./German-Traffic-Signs-Dataset-GTSRB-master/"

# Read the csv file that contains inside it train images address and label
train = pd.read_csv(path+"Train.csv")

# Adding images in 'train' DataFrame
images_list=[]
for i in train.Path:
    im = Image.open(path+i).resize((30,30)) # Opening images and Resize them
    images_list.append(np.array(im))

train["images"] = images_list
print("train's shape",train.images[0].shape)

print("train's info",train.info())

#Read the csv file that contains inside it test images address and label
test = pd.read_csv(path+"Test.csv")
images_list=[]
for i in test.Path:
    im = Image.open(path+i).resize((30,30)) # Opening images and Resize them
    images_list.append(np.array(im))
test["images"] = images_list
print(test.images[0].shape)

# Deleting variables that we no longer need
del images_list , i , im







