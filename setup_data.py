#!/usr/bin/python3
import h5py
import numpy as np

 # Construct dictionary of label, train and test datasets
 # from hdf5 files for network feeding. 
 # Final preprocessing steps: normalization 
 # and feature transformation

def get_data():
    f1 = h5py.File('images_train.hdf5', 'r')
    x_train = f1['x_train']
    x_val = f1['x_val']

    y_train =  f1['y_train']
    y_val = f1['y_val']

    f2 = h5py.File('images_test.hdf5', 'r')
    x_test = f2['x_test']

    def normalize(x):
        """
            argument
                - x: input image data in numpy array [300, 300]
            return
                - normalized x
        """
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x-min_val) / (max_val-min_val)
        return x

    x_train = normalize(x_train)
    x_val = normalize(x_val)
    x_test = normalize(x_test)

    y_train = np.stack(y_train)
    y_val = np.stack(y_val)


    data_dict = {
        'images_train': x_train,
        'labels_train': y_train,
        'images_val': x_val,
        'labels_val': y_val,
        'images_test': x_test,
    }
       
    return data_dict



def main():
  """Verify data structure is appropriate shape/type upon loading"""
  data_sets = get_data()
  print(data_sets['images_train'].shape)
  print(data_sets['labels_train'].shape)
  print(data_sets['images_val'].shape)
  print(data_sets['labels_val'].shape)
  print(data_sets['images_test'].shape)


if __name__ == '__main__':
    main()