#!/usr/bin/python3
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.ndimage as spimg
import numpy as np
import h5py
import tqdm, os, time
from img_parser import data, labels, files_test


np.set_printoptions(threshold=1000)

# Sample data/labels
x_train = data[0:int(len(data)*0.6)]
y_train = labels[0:int(len(labels)*0.6)]

x_val = data[int(len(data)*0.6):]
y_val = labels[int(len(labels)*0.6):]

x_test = files_test

# Use h5py to store large uncompressed image arrays to reduce memory requirements
def ImgRdr(arr, dataset):
    """Read images into scipy and convert to gray scale using generator function for memory efficiency."""
    slice = len(arr) # length of input arr
    files_convert = (spimg.imread(path, flatten=True) for path in arr[:slice])

    i = 0
    for r in tqdm.tqdm(files_convert, total=slice):
        # gray scale conversion
        r = r.astype('float32')/255
        # insert row into dataset
        dataset[i] = r
        i += 1

# Write hdf5 files (training data, test data) and store datasets on disk
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