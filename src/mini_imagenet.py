import os
import sys
import numpy as np
import pickle


def onehottify_2d_array(a):
    """
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
    :param a: 2-dimensional array.
    :return: 3-dim array where last dim corresponds to one-hot encoded vectors.
    """

    # https://stackoverflow.com/a/46103129/ @Divakar
    def all_idx(idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    num_columns = a.max() + 1
    out = np.zeros(a.shape + (num_columns,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out


class MiniImageNetData(object):

    def __init__(self, path, seed):
        """
        Constructs a miniImageNet dataset for use in episodic training.
        :param path: Path to miniImageNet data files.
        :param seed: Random seed to reproduce batches.
        """
        np.random.seed(seed)

        self.image_height = 84
        self.image_width = 84
        self.image_channels = 3

        path_train = os.path.join(path, 'mini_imagenet_train.pkl')
        path_validation = os.path.join(path, 'mini_imagenet_val.pkl')
        path_test = os.path.join(path, 'mini_imagenet_test.pkl')

        self.train_set = pickle.load(open(path_train, 'rb'))
        self.validation_set = pickle.load(open(path_validation, 'rb'))
        self.test_set = pickle.load(open(path_test, 'rb'))

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels

    def _sample_batch(self, images, tasks_per_batch, shot, way, eval_samples):
        """
        Sample a k-shot batch from images.
        :param images: Data to sample from [way, samples, h, w, c] (either of train, val, test)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: A list [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * (shot or eval_samples), h, w, c]
            * Labels: [tasks_per_batch, way * (shot or eval_samples), way]
                      (one-hot encoded in last dim)
        """

        samples_per_class = shot + eval_samples

        # Set up empty arrays
        train_images = np.empty((tasks_per_batch, way, shot, self.image_height, self.image_width,
                                self.image_channels), dtype=np.float32)
        test_images = np.empty((tasks_per_batch, way, eval_samples, self.image_height, self.image_width,
                                self.image_channels), dtype=np.float32)
        train_labels = np.empty((tasks_per_batch, way, shot), dtype=np.int32)
        test_labels = np.empty((tasks_per_batch, way, eval_samples), dtype=np.int32)

        classes_idx = np.arange(images.shape[0])
        samples_idx = np.arange(images.shape[1])

        # fill arrays one task at a time
        for i in range(tasks_per_batch):
            choose_classes = np.random.choice(classes_idx, size=way, replace=False)

            shape_imgs = images[choose_classes, :samples_per_class].shape
            imgs_tmp = np.zeros(shape_imgs)

            for j in range(way):
                choose_samples = np.random.choice(samples_idx, size=samples_per_class, replace=False)
                imgs_tmp[j, ...] = images[choose_classes[j], choose_samples, ...]

            labels_tmp = np.arange(way)

            train_images[i] = imgs_tmp[:, :shot].astype(dtype=np.float32)
            test_images[i] = imgs_tmp[:, shot:].astype(dtype=np.float32)

            train_labels[i] = np.expand_dims(labels_tmp, axis=1)
            test_labels[i] = np.expand_dims(labels_tmp, axis=1)

        # reshape arrays
        train_images = train_images.reshape(
            (tasks_per_batch, way * shot, self.image_height, self.image_width, self.image_channels)) / 255.
        test_images = test_images.reshape(
            (tasks_per_batch, way * eval_samples, self.image_height, self.image_width, self.image_channels)) / 255.
        train_labels = train_labels.reshape((tasks_per_batch, way * shot))
        test_labels = test_labels.reshape((tasks_per_batch, way * eval_samples))

        # labels to one-hot encoding
        train_labels = onehottify_2d_array(train_labels)
        test_labels = onehottify_2d_array(test_labels)

        return [train_images, test_images, train_labels, test_labels]

    def _shuffle_batch(self, train_images, train_labels):
        """
        Randomly permute the order of the second column
        :param train_images: [tasks_per_batch, way * shot, height, width, channels]
        :param train_labels: [tasks_per_batch, way * shot, way]
        :return: permuted images and labels.
        """
        for i in range(train_images.shape[0]):
            permutation = np.random.permutation(train_images.shape[1])
            train_images[i, ...] = train_images[i, permutation, ...]
            train_labels[i, ...] = train_labels[i, permutation, ...]
        return train_images, train_labels

    def get_batch(self, source, tasks_per_batch, shot, way, eval_samples):
        """
        Returns a batch of tasks from miniImageNet. Values are np.float32 and scaled to [0,1]
        :param source: one of `train`, `test`, `validation` (i.e. from which classes to pick)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * shot, height, width, channels]
            * Labels: [tasks_per_batch, way * shot, way]
                      (one-hot encoded in last dim)
        """

        # sample a batch
        if source == 'train':
            images = self.train_set
        elif source == 'validation':
            images = self.validation_set
        elif source == 'test':
            images = self.test_set

        train_images, test_images, train_labels, test_labels = self._sample_batch(images, tasks_per_batch, shot, way,
                                                                                  eval_samples)

        train_images, train_labels = self._shuffle_batch(train_images, train_labels)

        return [train_images, test_images, train_labels, test_labels]
