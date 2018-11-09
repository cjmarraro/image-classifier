# TensorFlow and Python Image Recognition 

## Description
This project covers the procedural steps for developing an image classification system. This includes downloading urls from csv files, parsing image files, preprocessesing images and creating hdf5 files to store relevent datasets, and building a predictive model in Tensorflow to assign labels to target images.

The images depict shoes from an online retail database, with each design/product id shown from multiple  viewpoints (between 9 and 13 views). Sideviews with sneakers' toes pointing to the right is the target. 

(Although csv files containing the image urls is not included in the repository due to privacy restrictions, a generic program for downloading and processing images is provided in the `load_url.py` module.)

After downloading tar files, unzip data: 

```bash
$ gunzip -c test_dat.tar.gz | tar xvf -
$ gunzip -c training_dat.tar.gz | tar xvf -
```

Training data filenames are prepended by either "0" or "1" to indicate label assignment, with "1" being the target label. 

A random sample of prediction results is found in the `img_prediction_dir` directory. 

```
Average accuracy >= 98%
```