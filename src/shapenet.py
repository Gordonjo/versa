import numpy as np
import sys
import os

"""
   Supporting methods for data handling
"""


def shuffle_batch(images, labels):
    """
       Return a shuffled batch of data
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]


def convert_index_to_angle(index, num_instances_per_item):
    """
    Convert the index of an image to a representation of the angle
    :param index: index to be converted
    :param num_instances_per_item: number of images for each item
    :return: a biterion representation of the angle
    """
    degrees_per_increment = 360./num_instances_per_item
    angle = index * degrees_per_increment
    angle_radians = np.deg2rad(angle)
    return angle, np.sin(angle_radians), np.cos(angle_radians)


class ShapeNetData(object):
    """
        Class to handle ShapeNet dataset. Loads from numpy data as saved in data folder.
    """
    def __init__(self, path, num_instances_per_item, train_fraction, val_fraction, seed, mode):
        """
        Initialize object to handle shapenet data
        :param path: directory of numpy file with preprocessed ShapeNet arrays.
        :param num_instances_per_item: Number of views of each model in the dataset.
        :param train_fraction: Fraction of models used for training.
        :param val_fraction: Fraction of models used for validation.
        :param seed: random seed for selecting data.
        :param mode: indicates either train or test.
        """
        self.image_height = 32
        self.image_width = 32
        self.image_channels = 1
        self.angle_dimensionality = 3
        self.has_validation_set = True

        # concatenate all the categories
        categories = ['02691156', '02828884', '02933112', '02958343', '02992529', '03001627', '03211117',
                      '03636649', '03691459', '04256520', '04379243', '04530566']
        for category in categories:
            file = os.path.join(path, '{0:s}.npy'.format(category))
            if category == categories[0]: # first time through
    	        data = np.load(file)
            else:
                data = np.concatenate((data, np.load(file)), axis=0)

        self.instances_per_item = num_instances_per_item
        self.total_items = data.shape[0]
        self.mode = mode
        train_size = (int) (train_fraction * self.total_items)
        val_size = (int) (val_fraction * self.total_items)
        print("Training Set Size = {0:d}".format(train_size))
        print("Validation Set Size = {0:d}".format(val_size))
        print("Test Set Size = {0:d}".format(self.total_items - train_size - val_size))
        np.random.seed(seed)
        np.random.shuffle(data)
        self.train_images, self.train_item_indices, self.train_item_angles = self.__extract_data(data[:train_size])
        self.validation_images, self.validation_item_indices, self.validation_item_angles = \
            self.__extract_data(data[train_size:train_size + val_size])
        self.test_images, self.test_item_indices, self.test_item_angles = \
            self.__extract_data(data[train_size + val_size:])
        self.train_item_sets = np.max(self.train_item_indices)
        self.validation_item_sets = np.max(self.validation_item_indices)
        self.test_item_sets = np.max(self.test_item_indices)
        if self.mode == 'test':
            self.test_item_permutation = np.random.permutation(self.test_item_sets)
            self.test_counter = 0

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels

    def get_angle_dimensionality(self):
        return self.angle_dimensionality

    def get_has_validation_set(self):
        return self.has_validation_set

    def get_batch(self, source, tasks_per_batch, shot):
        """
        Wrapper function for batching in the model.
        :param source: train, validation or test (string).
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :return: np array representing a batch of tasks.
        """
        if source == 'train':
            return self.__yield_random_task_batch(tasks_per_batch, self.train_images, self.train_item_angles,
                                                  self.train_item_indices, shot)
        elif source == 'validation':
            return self.__yield_random_task_batch(tasks_per_batch, self.validation_images, self.validation_item_angles,
                                                  self.validation_item_indices, shot)
        elif source == 'test':
            return self.__yield_random_task_batch(tasks_per_batch, self.test_images, self.test_item_angles,
                                                  self.test_item_indices, shot)

    def __yield_random_task_batch(self, num_tasks_per_batch, images, angles, item_indices, num_train_instances):
        """
        Generate a batch of tasks from image set.
        :param num_tasks_per_batch: number of tasks per batch.
        :param images: images set to generate batch from.
        :param angles: associated angle for each image.
        :param item_indices: indices of each character.
        :param num_train_instances: number of training images per class.
        :return: a batch of tasks.
        """
        train_images_to_return, test_images_to_return = [], []
        train_angles_to_return, test_angles_to_return = [], []
        for task in range(num_tasks_per_batch):
            images_train, images_test, labels_train, labels_test =\
                self.__generateRandomTask(images, angles, item_indices, num_train_instances)
            train_images_to_return.append(images_train)
            test_images_to_return.append(images_test)
            train_angles_to_return.append(labels_train)
            test_angles_to_return.append(labels_test)
        return np.array(train_images_to_return), np.array(test_images_to_return), \
               np.array(train_angles_to_return), np.array(test_angles_to_return)

    def __generateRandomTask(self, images, angles, item_indices, num_train_instances):
        """
        Randomly generate a task from image set.
        :param images: images set to generate batch from.
        :param angles: associated angle for each image.
        :param item_indices: indices of each character.
        :param num_train_instances: number of training images per class.
        :return: tuple containing train and test images and labels for a task.
        """
        if self.mode == 'test':
            task_item = self.test_item_permutation[self.test_counter]
            self.test_counter = self.test_counter + 1
        else:
            task_item = np.random.choice(np.unique(item_indices))
        permutation = np.random.permutation(self.instances_per_item)
        item_images = images[np.where(item_indices == task_item)[0]][permutation]
        item_angles = angles[np.where(item_indices == task_item)[0]][permutation]
        train_images, train_angles = item_images[:num_train_instances], item_angles[:num_train_instances]
        test_images, test_angles = item_images[num_train_instances:], item_angles[num_train_instances:]
        train_images_to_return, train_angles_to_return = shuffle_batch(train_images, train_angles)
        test_images_to_return, test_angles_to_return = shuffle_batch(test_images, test_angles)
        return train_images_to_return, test_images_to_return, train_angles_to_return, test_angles_to_return

    def __extract_data(self, data):
        """
        Unpack ShapeNet data.
        """
        images, item_indices, item_angles = [], [], []
        for item_index, item in enumerate(data):
            for m, instance in enumerate(item):
                images.append(instance[0])
                item_indices.append(item_index)
                item_angles.append(convert_index_to_angle(instance[2], self.instances_per_item))
        images = np.reshape(np.array(images), (len(images), self.image_height, self.image_width, self.image_channels))
        indices, angles = np.array(item_indices), np.array(item_angles)
        return images, indices, angles
