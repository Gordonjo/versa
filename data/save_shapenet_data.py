"""
save_shapenet_data.py:
    Script for saving the rendered ShapeNet images into a set of numpy files that are used as input in
    train_view_reconstruction.py and evaluate_view_reconstruction.py.
Usage instructions:
    1. In the data directory (the directory that this file resides in), create a new directory called shapenet.
    2. Download the shapenet dataset file shapenet_data.tar.gz from:
       https://drive.google.com/file/d/1y_-FcpKwPCOihizbQG0XqRxbg8lDUekz/view?usp=sharing
       and place it into the newly created shapenet directory.
    3. Decompress the dataset file (tar -xvzf shapenet_data.tar.gz) and place the contents in the shapenet directory.
    5. From the data directory, run this script as follows:
       python save_shapenet_data.py
    6. The result should be that the following twelve files are created in the data directory:
        02691156.npy
        02828884.npy
        02933112.npy
        02958343.npy
        02992529.npy
        03001627.npy
        03211117.npy
        03636649.npy
        03691459.npy
        04256520.npy
        04379243.npy
        04530566.npy

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


def load_pngs(input_dir, data, size):
    items = get_subdirs(input_dir)
    for item_index, item in enumerate(items):
        print(item)
        item_images = []
        instances = []
        # There are 36 generated orientations for each item
        for i in range(0, 36):
            instances.append("{0:02d}.png".format(i))

        for instance_index, instance in enumerate(instances):
            im = Image.open(os.path.join(item, instance))
            if size:
                im = im.resize((size, size), resample=Image.LANCZOS)
            image = np.array(im.getdata()).astype('float32').reshape(size, size) / 255.  # grayscale image
            item_images.append((image, item_index, instance_index))

        data.append(item_images)

    return data


def main():
    data_dir = './shapenet'
    categories = ['02691156', '02828884', '02933112', '02958343', '02992529', '03001627', '03211117', '03636649',
                  '03691459', '04256520', '04379243', '04530566']

    print('Starting...')
    for category in categories:
        data = []
        print('Working on category {0:s}'.format(category))
        data = load_pngs(os.path.join(data_dir, category), data, size=32)
        np.save('{0:s}.npy'.format(category), np.array(data))
    print('Finished!')


if __name__ == "__main__":
    main()
