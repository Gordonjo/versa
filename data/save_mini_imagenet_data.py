"""
Script for converting the raw miniImageNet data into .pkl files that are used as input in run_classifier.py.

--------------------------------------------------------------------------------
A portion of this code was derived from the MAML GitHub repository:
https://github.com/cbfinn/maml/blob/master/data/miniImagenet/proc_images.py
with the following notice:

Copyright (c) 2017 Chelsea Finn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
--------------------------------------------------------------------------------

Usage instructions:
1. From https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet,
   download train.csv, test.csv, and val.csv and place the 3 files in the data directory
   (the directory that this file resides in).
2. Acquire the minImageNet data files (images.zip) from Sachin Ravi at the email given in the paper:
   https://openreview.net/pdf?id=rJY0-Kcll.
3. In the data directory (the directory that this file resides in), create a new directory called mini_imagenet.
4. Put images.zip into the mini_imagenet directory and unzip it. The result will be a sub-directory
   of mini_imagenet called images that contains all the miniImageNet .jpg files.
4. From the data directory, run this script as follows:
    python save_mini_imagenet_data.py
5. The result should be that three .pkl files (mini_imagenet_train.pkl, mini_imagenet_test.pkl, mini_imagenet_val.pkl,)
   are created in the data directory.
"""

from __future__ import print_function
import numpy as np
import csv
import glob
import os
import pickle
from PIL import Image

def process_images(base_path):
    all_images = glob.glob(base_path + '/images/*')

    # Resize images
    print("Resizing images to 84 x 84 pixels.")
    for i, image_file in enumerate(all_images):
        im = Image.open(image_file)
        im = im.resize((84, 84), resample=Image.LANCZOS)
        im.save(image_file)
        if i % 500 == 0:
            print(i)
    print("Resizing complete.")

    # Put in correct directory
    print("Moving images to the train, test and val directories as specified in the .csv files.")
    for datatype in ['train', 'val', 'test']:
        os.system('mkdir ' + base_path + '/' + datatype)
        print("Moving {0:} images.".format(datatype))

        with open(datatype + '.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            last_label = ''
            for i, row in enumerate(reader):
                if i == 0:  # skip the headers
                    continue
                label = row[1]
                image_name = row[0]
                if label != last_label:
                    cur_dir = base_path + '/' + datatype + '/' + label + '/'
                    os.system('mkdir ' + cur_dir)
                    last_label = label
                os.system('mv ' + base_path + '/images/' + image_name + ' ' + cur_dir)
    print("Finished moving images.")


def save_file(base_path, data_name):
    print("Pickling {0:} images.".format(data_name))
    dir_list = os.listdir(os.path.join(base_path, data_name))
    dir_list = [os.path.join(base_path, data_name, x) for x in dir_list if os.path.isdir(os.path.join(base_path, data_name, x))]
    print(dir_list)
    output = np.zeros((len(dir_list), 600, 84, 84, 3), dtype=np.uint8)
    for i, dir in enumerate(dir_list):
        out = np.zeros((600, 84, 84, 3), dtype=np.uint8)
        im_files = glob.glob(os.path.join(dir, '*.jpg'))
        if len(im_files) != 600:
            print("Folder: {0:} should have 600 images in it and it has {1:d}"
                  .format(os.path.join(base_path, data_name, dir), len(im_files)))
            sys.exit("Problem with folder (%s)." % dir)
        for j, im_file in enumerate(im_files):
            im = Image.open(im_file)
            out[j] = im

        output[i] = out

    pickle.dump(output, open("mini_imagenet_" + data_name + ".pkl", 'wb'), protocol=2)


if __name__ == "__main__":

    base_path = './mini_imagenet'
    process_images(base_path)
    print("Pickling images.")
    save_file(base_path=base_path, data_name='train')
    save_file(base_path=base_path, data_name='val')
    save_file(base_path=base_path, data_name='test')
    print("Finished pickling images.")


