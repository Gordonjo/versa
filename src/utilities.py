from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io


"""
Probability Functions
"""


def sample_normal(mu, log_variance, num_samples):
    """
    Generate samples from a parameterized normal distribution.
    :param mu: tf tensor - mean parameter of the distribution.
    :param log_variance: tf tensor - log variance of the distribution.
    :param num_samples: np scalar - number of samples to generate.
    :return: tf tensor - samples from distribution of size num_samples x dim(mu).
    """
    shape = tf.concat([tf.constant([num_samples]), tf.shape(mu)], axis=-1)
    eps = tf.random_normal(shape, dtype=tf.float32)
    return mu + eps * tf.sqrt(tf.exp(log_variance))


def multinoulli_log_density(inputs, logits):
    """
    Compute the log density under a multinoulli distribution.
    :param inputs: tf tensor - inputs with axis -1 as random vectors.
    :param logits: tf tensor - logits parameterizing Bernoulli distribution.
    :return: tf tensor - log density under Multinoulli distribution.
    """
    return -tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits)


def gaussian_log_density(inputs, mu, logVariance):
    """
    Compute the log density under a parameterized normal distribution
    :param inputs: tf tensor - inputs with axis -1 as random vectors
    :param mu: tf tensor - mean parameter for normal distribution
    :param logVariance: tf tensor - log(sigma^2) of distribution
    :return: tf tensor - log density under a normal distribution
    """
    d = tf.cast(tf.shape(inputs)[-1], tf.float32)
    xc = inputs - mu
    return -0.5*(tf.reduce_sum((xc * xc) / tf.exp(logVariance), axis=-1)
                 + tf.reduce_sum(logVariance, axis=-1) + d * tf.log(2.0*np.pi))


"""
TensorFlow Network Support Functions
"""


def conv2d_pool_block(inputs, use_batch_norm, dropout_keep_prob, pool_padding, name):
    """
    A macro function that implements the following in sequence:
    - conv2d
    - batch_norm
    - relu activation
    - dropout
    - max_pool
    :param inputs: batch of feature maps.
    :param use_batch_norm: whether to use batch normalization or not.
    :param dropout_keep_prob: keep probability parameter for dropout.
    :param pool_padding: type of padding to use on the pooling operation.
    :param name: first part of the name used to scope this sequence of operations.
    :return: the processed batch of feature maps.
    """
    h = tf.layers.conv2d(
        inputs=inputs,
        strides=(1, 1),
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=xavier_initializer_conv2d(uniform=False),
        use_bias=False,
        name=(name + '_conv2d'),
        reuse=tf.AUTO_REUSE)

    if use_batch_norm:
        h = tf.contrib.layers.batch_norm(
            inputs=h,
            epsilon=1e-5,
            scope=(name + '_batch_norm'),
            reuse=tf.AUTO_REUSE)

    h = tf.nn.relu(features=h, name=(name + '_batch_relu'))

    h = tf.nn.dropout(x=h, keep_prob=dropout_keep_prob, name=(name + '_dropout'))

    h = tf.layers.max_pooling2d(inputs=h, pool_size=[2, 2], strides=2, padding=pool_padding, name=(name + '_pool'))

    return h


def dense_block(inputs, output_size, use_batch_norm, dropout_keep_prob, name):
    """
    A macro function that implements the following in sequence:
    - dense layer
    - batch_norm
    - relu activation
    - dropout
    :param inputs: batch of inputs.
    :param output_size: dimensionality of the output.
    :param use_batch_norm: whether to use batch normalization or not.
    :param dropout_keep_prob: keep probability parameter for dropout.
    :param name: first part of the name used to scope this sequence of operations.
    :return: batch of outputs.
    """
    h = tf.layers.dense(
        inputs=inputs,
        units=output_size,
        kernel_initializer=xavier_initializer(uniform=False),
        use_bias=False,
        name=(name + '_dense'),
        reuse=tf.AUTO_REUSE)

    if use_batch_norm:
        h = tf.contrib.layers.batch_norm(
            inputs=h,
            epsilon=1e-5,
            scope=(name + '_batch_norm'),
            reuse=tf.AUTO_REUSE)

    h = tf.nn.relu(features=h, name=(name + '_batch_relu'))

    h = tf.nn.dropout(x=h, keep_prob=dropout_keep_prob, name=(name + '_dropout'))

    return h


def dense_layer(inputs, output_size, activation, use_bias, name):
    """
    A simple dense layer.
    :param inputs: batch of inputs.
    :param output_size: dimensionality of the output.
    :param activation: activation function to use.
    :param use_bias: whether to have bias weights or not.
    :param name: name used to scope this operation.
    :return: batch of outputs.
     """
    return tf.layers.dense(
        inputs=inputs,
        units=output_size,
        kernel_initializer=xavier_initializer(uniform=False),
        use_bias=use_bias,
        bias_initializer=tf.random_normal_initializer(stddev=1e-3),
        activation=activation,
        name=name,
        reuse=tf.AUTO_REUSE)


def conv2d_transpose_layer(inputs, filters, activation, name):
    """
    A simple de-convolution layer.
    :param inputs: batch of inputs.
    :param filters: number of output filters.
    :param activation: activation function to use.
    :param name: name used to scope this operation.
    :return: batch of outputs.
     """
    return tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        activation=activation,
        data_format='channels_last',
        use_bias=False,
        kernel_initializer=xavier_initializer_conv2d(uniform=False),
        name=name,
        reuse=tf.AUTO_REUSE)



"""
print_and_log: Helper function to print to the screen and the log file.
"""


def print_and_log(log_file, message):
    print(message)
    log_file.write(message + '\n')


"""
get_log_files: Function that takes a path to a checkpoint directory and returns
a reference to a logfile and paths to the fully trained model and the model
with the best validation score.
"""


def get_log_files(checkpoint_dir):
    unique_dir_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    unique_checkpoint_dir = os.path.join(checkpoint_dir, unique_dir_name)
    if not os.path.exists(unique_checkpoint_dir):
        os.makedirs(unique_checkpoint_dir)
    checkpoint_path_validation = os.path.join(unique_checkpoint_dir, 'best_validation')
    checkpoint_path_final = os.path.join(unique_checkpoint_dir, 'fully_trained')
    logfile_path = os.path.join(unique_checkpoint_dir, 'log')
    logfile = open(logfile_path, "w")
    return logfile, checkpoint_path_validation, checkpoint_path_final


"""
plot_image_strips: Function to plot view reconstruction image strips.
"""


def plot_image_strips(shot_images, generated_images, generated_angles, ground_truth_images, ground_truth_angles, image_height, image_width, angles_to_plot, output_path):
    canvas_width = 14 # 1 shot image + 1 space + 12 images spaced every 30 degrees
    canvas_height = 2 # 1 row of generated images + 1 row of ground truth iamges
    canvas = np.empty((image_height * canvas_height, image_width * canvas_width))

    generated_images = np.array([generated_images[np.where(generated_angles[:, 0] == angle)[0]]
                        for angle in angles_to_plot[:, 0]]).squeeze(axis=1)
    ground_truth_images = np.array([ground_truth_images[np.where(ground_truth_angles[:, 0] == angle)[0]]
                            for angle in angles_to_plot[:, 0]]).squeeze(axis=1)
    # order images by angle
    generated_images = generated_images[np.argsort(angles_to_plot[:, 0], 0)]
    ground_truth_images = ground_truth_images[np.argsort(angles_to_plot[:, 0], 0)]

    blank_image = np.ones(shape=(image_height, image_width))

    # plot the first row which consists of: 1 shot image, 1 blank, 12 generated images equally spaced 30 degrees in azimuth
    # plot the shot image
    canvas[0:image_height, 0:image_width] = shot_images[0].squeeze()

    # plot 1 blank
    canvas[0:image_height, image_width:2 * image_width] = blank_image

    # plot generated images
    image_index = 0
    for column in range(2, canvas_width):
        canvas[0:image_height, column * image_width:(column + 1) * image_width] = generated_images[image_index].squeeze()
        image_index += 1

    # plot the ground truth strip in the 2nd row
    # Plot 2 blanks
    k = 0
    for column in range(0, 2):
        canvas[image_height:2 * image_height, column * image_width:(column + 1) * image_width] = blank_image
    # Plot ground truth images
    image_index = 0
    for column in range(2, canvas_width):
        canvas[image_height:2 * image_height, column * image_width:(column + 1) * image_width] = ground_truth_images[image_index].squeeze()
        image_index += 1

    plt.figure(figsize=(8, 10), frameon=False)
    plt.axis('off')
    plt.imshow(canvas, origin='upper', cmap='gray')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


"""
save_images_to_folder: Saves view reconstruction images to a folder.
"""


def save_images_to_folder(generated_images, generated_angles, ground_truth_images, ground_truth_angles, path):
    # order the images according to ascending angle
    ground_truth_images = ground_truth_images[np.argsort(ground_truth_angles[:, 0], 0)]
    generated_images = generated_images[np.argsort(generated_angles[:, 0], 0)]
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    counter = 0
    for (im_gt, im_gen) in zip(ground_truth_images, generated_images):
        ground_truth_path = os.path.join(path, 'ground_truth_{0:02d}.png'.format(counter))
        io.imsave(ground_truth_path, im_gt.squeeze())
        generated_path = os.path.join(path, 'generated_{0:02d}.png'.format(counter))
        io.imsave(generated_path, im_gen.squeeze())
        counter += 1
