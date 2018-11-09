#!/usr/bin/python3
import os
from random import shuffle

prefix_path_train = 'training_dat'
prefix_path_test = 'test_dat'

# Traverse filenames in image directory via list comprehension 
files_train = [os.path.join(prefix_path_train, name) for name in os.listdir(prefix_path_train)]
files_test = [os.path.join(prefix_path_test, name) for name in os.listdir(prefix_path_test)]

# One Hot Encoding
# Assign label 0 or 1 to corresponding image --> [('filename', [1,0] or [0,1])]. Target image := [0,1]
d=[]
for i in files_train:
    if i.split('/')[1].startswith('1'):
        d.append([i, [0.,1.]])
    else:
        d.append([i, [1.,0.]])

# Shuffle filename indices randomly w.r.t. labels. 
shuffle(d)
# Create data and label arrays 
data, labels = zip(*d)
del d

# Create test data array
shuffle(files_test)




                