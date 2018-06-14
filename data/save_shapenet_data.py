"""
save_shapenet_data.py:
    Script for saving the rendered ShapeNet images into a numpy file that is used as input in
    train_view_reconstruction.py and evaluate_view_reconstruction.py.
Usage instructions:
    1. Download the two shapenet dataset files 02691156.zip and 03001627.zip
       from https://drive.google.com/drive/folders/1x4EZFEE_bT9lvBu25ZnsMtV4LNKhYaG5?usp=sharing.
    2. In the data directory (the directory that this file resides in), create a new directory called shapenet.
    3. Unzip the two dataset files and place the conents in the shapenet directory.
    5. From the data directory, run this script as follows:
       python save_shapenet_data.py
    6. The result should be that a two files called shapenet_planes.npy and shapenet_chairs.npy
       are created in the data directory.

    Note: The created .npy files can only be used with the version of python that created it.
          In other words if you created the .npy file with python 2 it will not work with python 3
          or vice versa.
"""

import os
import sys
from PIL import Image
import numpy as np


def get_subdirs(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def load_pngs_and_save_as_npy(input_dir, save_file, model_type, size, convert_to_grayscale):
    data = []
    items = get_subdirs(input_dir)
    for item_index, item in enumerate(items):
        print(item)
        item_images = []
        # we have generated 36 orientations for each item
        # but they are numbered 1-9 (with a single digit),
        # and 10-35 (with 2 digits) and we want them in order
        instances = []
         # There are 36 generated orientations for each item
        if model_type == 'plane':
            # Planes  are numbered 1-9 (with a single digit), and 10-35 (with 2 digits) and we want them in order.
            for i in range(0, 10):
                instances.append("model_normalized.obj-{0:1d}.png".format(i))
            for i in range(10, 36):
                instances.append("model_normalized.obj-{0:2d}.png".format(i))
        elif model_type == 'chair':
            # Chairs are numbered consistently with 2 digits.
            for i in range(0, 36):
                instances.append("{0:02d}.png".format(i))
        else:
            sys.exit("Unsupported model type (%s)." % model_type)
            
        for instance_index, instance in enumerate(instances):
            im = Image.open(os.path.join(item, instance))
            if convert_to_grayscale:
                im = im.convert("L")
            if size:
                im = im.resize((size, size), resample=Image.LANCZOS)
            if convert_to_grayscale:
                image = np.array(im.getdata()).astype('float32').reshape(size, size) / 255. # grayscale image
            else:
                image = np.array(im.getdata()).astype('float32').reshape(size, size, 3) / 255. # colour image
            item_images.append((image, item_index, instance_index))

        data.append(item_images)

    np.save(save_file, np.array(data))


def main():
    data_dir = './shapenet'
    planes_dir = '02691156'
    chairs_dir = '03001627'

    print('Starting ShapeNet planes.')
    load_pngs_and_save_as_npy(os.path.join(data_dir, planes_dir), './shapenet_planes.npy', 
                              model_type='plane', size=32, convert_to_grayscale=True)
    print('Finished ShapeNet planes')

    print('Starting ShapeNet chairs.')
    load_pngs_and_save_as_npy(os.path.join(data_dir, chairs_dir), 'shapenet_chairs.npy',
                              model_type='chair', size=32, convert_to_grayscale=True)
    print('Finished ShapeNet chairs')


if __name__ == "__main__":
    main()
