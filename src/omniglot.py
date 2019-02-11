import numpy as np

"""
   Supporting methods for data handling
"""


def shuffle_batch(images, labels):
    """
       Return a shuffled batch of data
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]


def extract_data(data, augment_data):
    images, char_nums = [], []
    if augment_data:
        for character in data:
            data = augment_character_set(data, character)
    for character_index, character in enumerate(data):
        for m, instance in enumerate(character):
            images.append(instance[0])
            char_nums.append(character_index)
    images = np.expand_dims(np.array(images), 4)
    char_number = np.array(char_nums)
    return images, char_number


def augment_character_set(data, character_set):
    """
    :param data: Dataset the character belongs to.
    :param character_set: np array containing instances of a character.
    :return: Original data with added character sets for all defined permutations of the current character.
    """
    rotation_90, rotation_180, rotation_270 = [], [], []
    for instance in character_set:
        image, char_num, char_language_num = instance
        rotation_90.append((np.rot90(image, k=1), char_num, char_language_num))
        rotation_180.append((np.rot90(image, k=2), char_num, char_language_num))
        rotation_270.append((np.rot90(image, k=3), char_num, char_language_num))
    return np.vstack((data, np.array([rotation_90, rotation_180, rotation_270])))


class OmniglotData(object):
    """
        Class to handle Omniglot data set. Loads from numpy data as saved in
        data folder.
    """
    def __init__(self, path, train_size, validation_size, augment_data, seed):
        """
        Initialize object to handle Omniglot data
        :param path: directory of numpy file with preprocessed Omniglot arrays.
        :param train_size: Number of characters in training set.
        :param validation_size: Number of characters in validation set.
        :param augment_data: Augment with rotations of characters (boolean).
        :param seed: random seed for train/validation/test split.
        """
        np.random.seed(seed)

        data = np.load(path)
        np.random.shuffle(data)

        self.instances_per_char = 20
        self.image_height = 28
        self.image_width = 28
        self.image_channels = 1
        self.total_chars = data.shape[0]

        self.train_images, self.train_char_nums = extract_data(data[:train_size], augment_data=augment_data)
        if validation_size is not 0:
            self.validation_images, self.validation_char_nums =\
                extract_data(data[train_size:train_size + validation_size], augment_data=augment_data)
        self.test_images, self.test_char_nums =\
            extract_data(data[train_size + validation_size:], augment_data=augment_data)

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels

    def get_batch(self, source, tasks_per_batch, shot, way, eval_samples):
        """
        Gets a batch of data.
        :param source: train, validation or test (string).
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: np array representing a batch of tasks.
        """
        if source == 'train':
            return self._yield_random_task_batch(tasks_per_batch, self.train_images, self.train_char_nums, shot, way, eval_samples)
        elif source == 'validation':
            return self._yield_random_task_batch(tasks_per_batch, self.validation_images, self.validation_char_nums,
                                                  shot, way, eval_samples)
        elif source == 'test':
            return self._yield_random_task_batch(tasks_per_batch, self.test_images, self.test_char_nums, shot, way, eval_samples)

    def _yield_random_task_batch(self, tasks_per_batch, images, character_indices, shot, way, eval_samples):
        """
        Generate a batch of tasks from image set.
        :param tasks_per_batch: Number of tasks per batch.
        :param images: Images set to generate batch from.
        :param character_indices: Index of each character.
        :param shot: Number of training images per class.
        :param way: Number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: A batch of tasks.
        """
        train_images_to_return, test_images_to_return = [], []
        train_labels_to_return, test_labels_to_return = [], []
        for task in range(tasks_per_batch):
            im_train, im_test, lbl_train, lbl_test = self._generate_random_task(images, character_indices, shot, way,
                                                                                eval_samples)
            train_images_to_return.append(im_train)
            test_images_to_return.append(im_test)
            train_labels_to_return.append(lbl_train)
            test_labels_to_return.append(lbl_test)
        return np.array(train_images_to_return), np.array(test_images_to_return),\
               np.array(train_labels_to_return), np.array(test_labels_to_return)

    def _generate_random_task(self, images, character_indices, shot, way, eval_samples):
        """
        Randomly generate a task from image set.
        :param images: images set to generate batch from.
        :param character_indices: indices of each character.
        :param shot: number of training images per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: tuple containing train and test images and labels for a task.
        """
        num_test_instances = eval_samples
        train_images_list, test_images_list = [], []
        task_characters = np.random.choice(np.unique(character_indices), way)
        for character in task_characters:
            character_images = images[np.where(character_indices == character)[0]]
            np.random.shuffle(character_images)
            train_images_list.append(character_images[:shot])
            test_images_list.append(character_images[shot:shot + eval_samples])
        train_images_to_return, test_images_to_return = np.vstack(train_images_list), np.vstack(test_images_list)
        train_labels_to_return = np.eye(way).repeat(shot, 0)
        test_labels_to_return = np.eye(way).repeat(num_test_instances, 0)
        train_images_to_return, train_labels_to_return = shuffle_batch(train_images_to_return, train_labels_to_return)
        test_images_to_return, test_labels_to_return = shuffle_batch(test_images_to_return, test_labels_to_return)
        return train_images_to_return, test_images_to_return, train_labels_to_return, test_labels_to_return
