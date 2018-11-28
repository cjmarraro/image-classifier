# TensorFlow and Python Image Recognition 

## Description
Components for developing an image classification system include: web scraping module for loading images, parsing image files, image preprocessing and hdf5 file creation to efficiently store relevent datasets, and building a predictive model in Tensorflow to assign labels to test images.

The images depict shoes and apparel from an online retail database, with each style/product displayed at various angles (9 to 13 views per product). Sideviews of sneakers' toes pointing to the right is the target view in this experiment. 

(Although csv files containing image urls are not included in the repository due to privacy restrictions, a basic program for downloading and preprocessing images is provided in the `load_url.py` module. Here, training data filenames are prepended by either "0" or "1" to indicate preassigned labels, "1" being the target label, which supports subsequent file parsing procedures).

After downloading tar files, unzip data: 

```bash
$ gunzip -c test_dat.tar.gz | tar xvf -
$ gunzip -c training_dat.tar.gz | tar xvf -
```

A random selection of prediction results is found in the `img_prediction_dir` directory. 

```
Average accuracy >= 99%
```