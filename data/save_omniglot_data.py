"""
save_omniglot_data.py:
    Script for saving the raw Omniglot data into a numpy file that is used as input in run_classifier.py.
Usage instructions:
    1. Download the two omniglot dataset files images_background.zip and images_evaluation.zip
       from https://github.com/brendenlake/omniglot/tree/master/python.
    2. Unzip the two dataset files.
    3. In the data directory (the directory that this file resides in), create a new directory called omniglot.
    4. Move the contents of the two unzip files .zip files (not including the images_background and
       images_evaluation root folders) into the omniglot directory created in the previous step. The omniglot
       directory should now contain 50 language directories.
    5. From the data directory, run this script as follows:
       python save_omniglot_data.py
    6. The result should be that a file called omniglot.npy is created in the data directory.

    Note: The created omniglot.npy file can only be used with the version of python that created it.
          In other words if you created the .npy file with python 2 it will not work with python 3
          or vice versa.
"""

import os
from PIL import Image
import numpy as np

data_dir = '.'

def get_subdirs(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def load_and_save(save_file, size=None):
    data = []
    languages = get_subdirs(os.path.join(data_dir, 'omniglot'))

    for language_num, language in enumerate(languages):
        characters = get_subdirs(language)
        characters.sort()
        for character_num, character in enumerate(characters):
            character_images = []
            instances = os.listdir(character)
            instances.sort()
            for instance in instances:
                im = Image.open(os.path.join(character, instance))
                if size:
                    im = im.resize((size, size), resample=Image.LANCZOS)
                image = np.array(im.getdata()).astype('float32').reshape(size, size) / 255.
                image = 1.0 - image  # invert the data as Omniglot is black on white

                character_images.append((image, character_num, language_num))
            data.append(character_images)

    np.save(save_file, np.array(data))


def main():
    print('Started Omniglot data preparation.')
    load_and_save(os.path.join(data_dir, 'omniglot.npy'), size=28)
    print('Finished. Omniglot data saved as omniglot.npy.')


if __name__ == "__main__":
    main()

