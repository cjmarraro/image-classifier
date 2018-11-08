# TensorFlow and Python Image Recognition 

## Description
This project covers the procedural steps for developing an image classification system. This includes downloading and parsing image urls from csv files, preprocessesing images, creating hdf5 files to store relevent datasets, and building a predictive model in Tensorflow to assign labels to target images.

The images depict sneakers from an online retail database, with each design/product id displayed with multiple viewpoints. Sideview images with sneakers' toes pointing to the right is the target. 

(Although csv files containing the image urls is not included in the repository due to privacy restrictions, a generic program for downloading and processing images is provided in the `import_imgs_from_url.py` module.)

To download image data: 

```bash
$ gunzip -c test_dat.tar.gz | tar xvf -
$ gunzip -c training_dat.tar.gz | tar xvf -
```

Training data filenames are prepended by either "0" or "1" to indicate label assignment, with "1" being the target label. 

A random sampling of prediction results is found in the `img_prediction_dir` directory. 

```
Average accuracy >= 98%
```