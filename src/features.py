import tensorflow as tf
from utilities import conv2d_pool_block, conv2d_transpose_layer, dense_layer, dense_block


def extract_features_shapenet(images, output_size, use_batch_norm, dropout_keep_prob):
    """
    Based on the architecture described in 'Matching Networks for One-Shot Learning'
    http://arxiv.org/abs/1606.04080.pdf.
    :param images: batch of images.
    :param output_size: dimensionality of the output features.
    :param use_batch_norm: whether to use batch normalization or not.
    :param dropout_keep_prob: keep probability parameter for dropout.
    :return: features.
    """

    # 4X conv2d + pool blocks
    h = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'same','fe_block_1')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same','fe_block_2')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same','fe_block_3')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_4')

    # flatten output
    h = tf.contrib.layers.flatten(h)

    # dense layer
    h = dense_block(h, output_size, use_batch_norm, dropout_keep_prob, 'fe_dense')

    return h


def extract_features_omniglot(images, output_size, use_batch_norm, dropout_keep_prob):
    """
    Based on the architecture described in 'Matching Networks for One-Shot Learning'
    http://arxiv.org/abs/1606.04080.pdf.

    :param images: batch of images.
    :param output_size: dimensionality of the output features.
    :param use_batch_norm: whether to use batch normalization or not.
    :param dropout_keep_prob: keep probability parameter for dropout.
    :return: features.
    """

    # 4X conv2d + pool blocks
    h = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_1')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_2')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_3')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_4')

    # flatten output
    h = tf.contrib.layers.flatten(h)

    return h


def extract_features_mini_imagenet(images, output_size, use_batch_norm, dropout_keep_prob):
    """
    Based on the architecture described in 'Matching Networks for One-Shot Learning'
    http://arxiv.org/abs/1606.04080.pdf.

    :param images: batch of images.
    :param output_size: dimensionality of the output features.
    :param use_batch_norm: whether to use batch normalization or not.
    :param dropout_keep_prob: keep probability parameter for dropout.
    :return: features.
    """

    # 5X conv2d + pool blocks
    h = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_1')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_2')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_3')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_4')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_5')

    # flatten output
    h = tf.contrib.layers.flatten(h)

    return h


def generate_views(angles, adaptation_inputs):
    """
    Based on the architecture described in 'Matching Networks for One-Shot Learning'
    http://arxiv.org/abs/1606.04080.pdf.

    :param angles: batch of orientationses.
    :param adaptation_inputs: batch of adaptation_inputs.
    :return: batch of generated views.
    """

    h = tf.concat([angles, adaptation_inputs], -1)

    h = dense_layer(inputs=h, output_size=512, activation=tf.nn.relu, use_bias=False, name='generate_dense_1')
    h = dense_layer(inputs=h, output_size=1024, activation=tf.nn.relu, use_bias=False, name='generate_dense_2')

    h = tf.reshape(h, shape=[-1, 2, 2, 256])

    h = conv2d_transpose_layer(inputs=h, filters=128, activation=tf.nn.relu, name='generate_deconv_1')
    h = conv2d_transpose_layer(inputs=h, filters=64, activation=tf.nn.relu, name='generate_deconv_2')
    h = conv2d_transpose_layer(inputs=h, filters=32, activation=tf.nn.relu, name='generate_deconv_3')
    h = conv2d_transpose_layer(inputs=h, filters=1, activation=tf.nn.sigmoid, name='generate_deconv_4')

    return h
