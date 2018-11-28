#!/usr/bin/python3
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.ndimage as spimg
import numpy as np
import h5py
import tqdm, os, time
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


np.set_printoptions(threshold=1000)

# Sample data and label sets
x_train = data[0:int(len(data)*0.6)]
y_train = labels[0:int(len(labels)*0.6)]

x_val = data[int(len(data)*0.6):]
y_val = labels[int(len(labels)*0.6):]

x_test = files_test

# Use h5py to store large uncompressed image arrays to reduce memory requirements
def ImgRdr(arr, dataset):  
    """Function for reading images into scipy and flattening 
     to grayscale. Use generator object for memory efficiency."""   
    slice = len(arr) # length of input array
    files_convert = (spimg.imread(path, flatten=True) for path in arr[:slice])

    i = 0
    for r in tqdm.tqdm(files_convert, total=slice):
        # grayscale conversion: [0,255] --> [0,1]
        r = r.astype('float32')/255
        # insert row into dataset
        dataset[i] = r
        i += 1

# Write hdf5 files (training data, test data) to store datasets on disk
dir = os.getcwd()
for k in ['images_train.hdf5', "images_test.hdf5"]:
    if not os.path.exists(os.path.join(dir, k)):
        f1 = h5py.File('images_train.hdf5', 'w')
        
        #training images
        f1.create_dataset('x_train', (len(x_train), 300, 300))
        ImgRdr(x_train, f1['x_train']) 
        
        #validation images
        f1.create_dataset('x_val', (len(x_val), 300, 300))
        ImgRdr(x_val, f1['x_val']) 

        #training and validation labels
        f1.create_dataset('y_train', data=np.array(y_train))
        f1.create_dataset('y_val', data=np.array(y_val))
       
        f2 = h5py.File('images_test.hdf5', 'w')
        
        #test images
        f2.create_dataset('x_test', (len(x_test), 300, 300))
        ImgRdr(x_test, f2['x_test'])

f1.close()
f2.close()

if __name__=='__main__':
    start = time.time()

    os.system("tar cvf - training_dat | gzip > training_dat.tar.gz")
    os.system("tar cvf - test_dat | gzip > test_dat.tar.gz")


    end = time.time()
    print("Elapsed time: %s"%(end - start,))