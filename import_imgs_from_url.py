#!/usr/bin/env python3
import sys, os, multiprocessing, csv
from PIL import Image
from io import BytesIO
import requests


def image_indexing(data_file):
    images = []
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        dataset = list(reader)
        
        for col in dataset:
            if len(col) == 3:
                images.append([col[0], col[1], col[2]])
            else:
                images.append([col[0], col[1]])
                
        return images[1:]

def download_image(key_url):
    out_dir = sys.argv[2]
    while len(key_url[0]) == 3:
        (key, url, labels) = key_url
        filename = os.path.join(out_dir, str(labels)+'_'+'{}'.format(key)+'.jpeg')
        
        break 
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}'.format(key)+'.jpeg')
     
    if os.path.exists(filename):
        
        print('Image {} already exists. Skipping download.'.format(filename))
        return 
    
    try:
        response = requests.get(url)
        image_data = response.content 
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return
    
    try:
        pil_image = Image.Image.resize(pil_image, (300,300), Image.ANTIALIAS)
    except:
        print('Warning: Failed to resize image {}'.format(key))
        return

    try:
        pil_image_RGB = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return

    
    try:
        pil_image_RGB.save(filename, format='JPEG', quality=90, optimize=True)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return



def loader():
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = image_indexing(data_file)
    pool = multiprocessing.Pool(processes=4)  # Num of CPUs
    pool.map(download_image, key_url_list)
    pool.close()
    pool.terminate()

if __name__ == '__main__':
    loader()
    
        