# TensorFlow and Python Image Recognition 

## Description
The project provides components for developing an image classification system. This includes web scraping image urls for downloading, parsing image files, image preprocessing and creating hdf5 files to store relevent datasets, and building a predictive model in Tensorflow to assign labels to test images.

The images are of shoes from an online retail database, with each design/product displayed at various angles (9 to 13 views per design). Sideviews of sneakers' toes pointing to the right is the target. 

(Although csv files containing image urls are not included in the repository due to privacy restrictions, a basic program for downloading and processing images is provided in the `load_url.py` module. Here, training data filenames are prepended by either "0" or "1" to indicate preassigned labels, "1" being the target label,which supports subsequent file parsing methods).

After downloading tar files, unzip data: 

```bash
$ gunzip -c test_dat.tar.gz | tar xvf -
$ gunzip -c training_dat.tar.gz | tar xvf -
```

A random selection of prediction results is found in the `img_prediction_dir` directory. 

```
Average accuracy >= 98%
```