import tensorflow as tf
from utilities import dense_layer, sample_normal


def infer_classifier(features, labels, d_theta, num_classes):
    """
    Infer a linear classifier by concatenating vectors for each class.
    :param features: tensor (tasks_per_batch x num_features) feature matrix
    :param labels:  tensor (tasks_per_batch x num_classes) one-hot label matrix
    :param d_theta: Integer number of features on final layer before classifier.
    :param num_classes: Integer number of classes per task.
    :return: Dictionary containing output classifier layer (including means and
    :        log variances for weights and biases).
    """
    # dense layer before pooling
    pre_pooling = dense_layer(features, d_theta, None, True, 'pre_process_dense')

    classifier = {}
    class_weight_means = []
    class_weight_logvars = []
    class_bias_means = []
    class_bias_logvars = []
    for c in range(num_classes):
        class_mask = tf.equal(tf.argmax(labels, 1), c)
        class_features = tf.boolean_mask(pre_pooling, class_mask)
        
        # Pool across dimensions
        nu = tf.expand_dims(tf.reduce_mean(class_features, axis=0), axis=0)
        post_processed = __post_process(nu, d_theta)

        class_weight_means.append(dense_layer(post_processed, d_theta, None, True, 'weight_mean'))
        class_weight_logvars.append(dense_layer(post_processed, d_theta, None, True, 'weight_log_variance'))
        class_bias_means.append(dense_layer(post_processed, 1, None, True, 'bias_mean'))
        class_bias_logvars.append(dense_layer(post_processed, 1, None, True, 'bias_log_variance'))

    classifier['weight_mean'] = tf.transpose(tf.concat(class_weight_means, axis=0))
    classifier['bias_mean'] = tf.reshape(tf.concat(class_bias_means, axis=1), [num_classes, ])
    classifier['weight_log_variance'] = tf.transpose(tf.concat(class_weight_logvars, axis=0))
    classifier['bias_log_variance'] = tf.reshape(tf.concat(class_bias_logvars, axis=1), [num_classes, ])
    
    return classifier


def shapenet_inference(image_features, angles, d_theta, d_psi, num_samples):
    """
    Perform inference for the view reconstruction task
    :param image_features: tensor (N x d) of training image features.
    :param angles: tensor (N x d_angle) of training image angles.
    :param d_theta: integer dimensionality of the features.
    :param d_psi: integer dimensionality of adaptation parameters.
    :param num_samples: number of samples to generate from the distribution.
    :return: dictionary containing distribution parameters and samples.
    """
    # Concatenate features and angles
    h = tf.concat([image_features, angles], axis=-1)

    # denses layer before pooling
    h = dense_layer(inputs=h, output_size=d_theta, activation=tf.nn.elu, use_bias=True, name='pre_process_dense_1')
    h = dense_layer(inputs=h, output_size=d_theta, activation=tf.nn.elu, use_bias=True, name='pre_process_dense_2')

    # Pool across dimensions
    nu = tf.expand_dims(tf.reduce_mean(h, axis=0), axis=0)
    post_processed = __post_process(nu, d_psi)

    # Compute means and log variances for the parameter
    psi = {}
    psi['mu'] = dense_layer(inputs=post_processed, output_size=d_psi, activation=None, use_bias=True, name='psi_mean')
    psi['log_variance'] = \
        dense_layer(inputs=post_processed, output_size=d_psi, activation=None, use_bias=True, name='psi_log_var')

    psi['psi_samples'] = sample_normal(psi['mu'], psi['log_variance'], num_samples)
    return psi


def __post_process(pooled, units):
    """
    Process a pooled variable through 2 dense layers
    :param pooled: tensor of rank (1 x num_features).
    :param units: integer number of output features.
    :return: tensor of rank (1 x units)
    """
    h = dense_layer(pooled, units, tf.nn.elu, True, 'post_process_dense_1')
    h = dense_layer(h, units, tf.nn.elu, True, 'post_process_dense_2')

    return h
