"""
Script to reproduce the few-shot view reconstruction results in:
"Meta-Learning Probabilistic Inference For Prediction"
https://arxiv.org/pdf/1805.09921.pdf

The following command lines will reproduce the published results within error-bars:

python train_view_reconstruction.py
python evaluate_view_reconstruction.py -m ./checkpoint/<date-time>/fully_trained

where <date-time> is the specific time stamped folder where train_view_reconstruction.py saves the model.

"""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import argparse
from features import extract_features_shapenet, generate_views
from inference import shapenet_inference
from utilities import print_and_log, get_log_files, gaussian_log_density
from data import get_data

"""
parse_command_line: command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_theta", type=int, default=256, help="Shared parameters dimensionality.")
    parser.add_argument("--d_psi", type=int, default=256, help="Adaptation input dimensionality.")
    parser.add_argument("--shot", type=int, default=1, help="Number of training examples.")
    parser.add_argument("--tasks_per_batch", type=int, default=24, help="Number of tasks per batch.")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples from q.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--iterations", type=int, default=500000, help="Number of training iterations.")
    parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint', help="Directory to save trained models.")
    parser.add_argument("--random_shot", default=False, action="store_true", help="Randomize the shot between 1 and shot.")
    parser.add_argument("--print_freq", type=int, default=200, help="Frequency of summary results (in iterations).")

    args = parser.parse_args()

    return args


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = parse_command_line()

    logfile, checkpoint_path_validation, checkpoint_path_final = get_log_files(args.checkpoint_dir)

    print_and_log(logfile, "Options: %s\n" % args)

    # Load training and eval data
    data = get_data("shapenet")

    # tf placeholders
    batch_train_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                                     None,  # shot
                                                     data.get_image_height(),
                                                     data.get_image_width(),
                                                     data.get_image_channels()], name='train_images')
    batch_test_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                                    None,  # num test images
                                                    data.get_image_height(),
                                                    data.get_image_width(),
                                                    data.get_image_channels()], name='test_images')
    batch_train_angles = tf.placeholder(tf.float32, [None,  # tasks per batch
                                                     None,  # shot
                                                     data.get_angle_dimensionality()], name='train_angles')
    batch_test_angles = tf.placeholder(tf.float32, [None,  # tasks per batch
                                                    None,  # num test angles
                                                    data.get_angle_dimensionality()], name='test_angles')

    def evaluate_task(inputs):
        train_images, train_angles, test_images, test_angles = inputs
        inference_features_train = extract_features_shapenet(images=train_images, output_size=args.d_theta,
                                                    use_batch_norm=False, dropout_keep_prob=1.0)
        adaptation_params = shapenet_inference(inference_features_train, train_angles, args.d_theta,
                                               args.d_psi, args.samples)
        test_batch_size = tf.shape(test_images)[0]
        sample_log_py = []

        # loop over samples
        for n in range(args.samples):
            adaptation_vector = adaptation_params['psi_samples'][n, :, :]
            adaptation_inputs = tf.tile(adaptation_vector, [test_batch_size, 1])
            generated_images = generate_views(test_angles, adaptation_inputs)
            # Compute loss
            flat_images_gt = tf.reshape(test_images,
                                        [-1,
                                         data.get_image_height() * data.get_image_width() * data.get_image_channels()])
            flat_images_gen = tf.reshape(generated_images,
                                         [-1,
                                          data.get_image_height() * data.get_image_width() * data.get_image_channels()])
            log_var = tf.zeros_like(flat_images_gt)
            log_density = gaussian_log_density(flat_images_gt, flat_images_gen, log_var)
            sample_log_py.append(tf.expand_dims(log_density, 1))
        task_log_py = tf.reduce_mean(tf.concat(sample_log_py, 1), axis=1)
        task_loss = -task_log_py
        return [task_loss, task_log_py]

    # tf mapping of batch to evaluation function
    batch_output = tf.map_fn(fn=evaluate_task,
                             elems=(batch_train_images, batch_train_angles, batch_test_images, batch_test_angles),
                             dtype=[tf.float32, tf.float32],
                             parallel_iterations=args.tasks_per_batch)

    # average all values across batch
    batch_losses, batch_log_densities = batch_output
    loss = tf.reduce_mean(batch_losses)
    log_likelihood = tf.reduce_mean(batch_log_densities)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    gvs = optimizer.compute_gradients(loss)
    gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs if grad is not None]
    train_step = optimizer.apply_gradients(gvs)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        # train the model
        validation_batches = 100
        iteration = 0
        best_validation_loss = 5e10
        train_iteration_loss = []
        sess.run(tf.global_variables_initializer())
        while iteration < args.iterations:
            train_shot = args.shot
            if (args.random_shot):
                train_shot = np.random.randint(low=1, high=(args.shot + 1))
            train_inputs, test_inputs, train_outputs, test_outputs = \
                data.get_batch(source='train', tasks_per_batch=args.tasks_per_batch, shot=train_shot)
            feed_dict = {batch_train_images: train_inputs, batch_test_images: test_inputs,
                         batch_train_angles: train_outputs, batch_test_angles: test_outputs}
            _, log_py, iteration_loss = sess.run([train_step, log_likelihood, loss], feed_dict)
            train_iteration_loss.append(iteration_loss)
            if (iteration > 0) and (iteration % args.print_freq == 0):
                validation_iteration, iteration_loss = 0, []
                while validation_iteration < validation_batches:
                    train_inputs, test_inputs, train_outputs, test_outputs = \
                        data.get_batch(source='validation', tasks_per_batch=args.tasks_per_batch, shot=args.shot)
                    feed_dict = {batch_train_images: train_inputs, batch_test_images: test_inputs,
                                 batch_train_angles: train_outputs, batch_test_angles: test_outputs}
                    iter_loss = sess.run([loss], feed_dict)
                    iteration_loss.append(iter_loss)
                    validation_iteration += 1
                validation_loss = np.array(iteration_loss).mean()
                train_average_loss = np.array(train_iteration_loss).mean()

                # save checkpoint if validation is the best so far
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    saver.save(sess=sess, save_path=checkpoint_path_validation)

                print_and_log(logfile, 'Iteration: {}, Likelihood: {:5.3f}, Iteration-Train-Loss: {:5.3f},'
                                       'Val-Loss: {:5.3f}'.format(iteration, log_py, train_average_loss,
                                                                  validation_loss))
                train_iteration_loss = []

            iteration += 1
        # save the checkpoint from the final epoch
        saver.save(sess, save_path=checkpoint_path_final)
        print_and_log(logfile, 'Fully-trained model saved to: {}'.format(checkpoint_path_final))
        print_and_log(logfile, 'Best validation loss: {:5.3f}'.format(best_validation_loss))
        print_and_log(logfile, 'Best validation model saved to: {}'.format(checkpoint_path_validation))

    logfile.close()


if __name__ == "__main__":
    tf.app.run()