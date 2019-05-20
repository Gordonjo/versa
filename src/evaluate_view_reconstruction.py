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
from utilities import plot_image_strips, save_images_to_folder, gaussian_log_density
from data import get_data
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_ssim as ssim
import math
import os


def compute_image_quality_metrics(ground_truth_images, ground_truth_angles, generated_images, generated_angles):
    # order the images according to ascending angle
    ground_truth_images = ground_truth_images[np.argsort(ground_truth_angles[:, 0], 0)]
    generated_images = generated_images[np.argsort(generated_angles[:, 0], 0)]

    loop_mse, loop_nrmse, loop_ssim = [], [], []
    for (im_gt, im_gen) in zip(ground_truth_images, generated_images):
        loop_mse.append(mse(im_gt, im_gen))
        loop_nrmse.append(nrmse(im_gt, im_gen))
        loop_ssim.append(ssim(im_gt.squeeze(), im_gen.squeeze()))
    return np.array(loop_mse).mean(), np.array(loop_nrmse).mean(), np.array(loop_ssim).mean()


"""
parse_command_line: command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_theta", type=int, default=256, help="Shared parameters dimensionality.")
    parser.add_argument("--d_psi", type=int, default=256, help="Adaptation input dimensionality.")
    parser.add_argument("--shot", type=int, default=1, help="Number of training samples.")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples from q.")
    parser.add_argument("--iterations", type=int, default=7420, help="Number of test iterations.")
    parser.add_argument("--model_path", "-m", default=None, help="Model to load and test.")

    args = parser.parse_args()

    return args


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = parse_command_line()

    # Load training and eval data
    data = get_data("shapenet", mode='test')

    test_tasks_per_batch = 1  # always use a batch size of 1 for testing
    test_shot = args.shot
    test_iteration = 0
    task_bits_per_pixel = []
    task_mse = []
    task_nrmse = []
    task_ssim = []
    plot_angles = np.arange(0, 360, 30)  # plot every 30 degrees in azimuth
    angles_to_plot = np.array([plot_angles, np.sin(np.deg2rad(plot_angles)), np.cos(np.deg2rad(plot_angles))]).T
    generate_angles = np.arange(0, 360, 10)  # ask the model to generate views every 10 degrees in azimuth
    angles_to_generate = np.array(
        [generate_angles, np.sin(np.deg2rad(generate_angles)), np.cos(np.deg2rad(generate_angles))]).T
    all_angles = np.tile(np.expand_dims(angles_to_generate, 0), (test_tasks_per_batch, 1, 1))
    while test_iteration < args.iterations:
        # tf placeholders
        batch_train_images = tf.placeholder(tf.float32,
                                            [None,  # tasks per batch
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
        batch_all_angles = tf.placeholder(tf.float32, [None,  # tasks per batch
                                                       None,  # num test angles + num train angles
                                                       data.get_angle_dimensionality()], name='all_angles')

        def evaluate_task(inputs):
            train_images, train_angles, test_images, test_angles, all_batch_angles = inputs
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
                gen_images = generate_views(test_angles, adaptation_inputs)
                # Compute loss
                flat_images_gt = tf.reshape(test_images,
                                            [-1,
                                             data.get_image_height() * data.get_image_width() * data.get_image_channels()])
                flat_images_gen = tf.reshape(gen_images,
                                             [-1,
                                              data.get_image_height() * data.get_image_width() * data.get_image_channels()])
                log_var = tf.zeros_like(flat_images_gt)
                log_density = gaussian_log_density(flat_images_gt, flat_images_gen, log_var)
                sample_log_py.append(tf.expand_dims(log_density, 1))
            task_log_py = tf.reduce_logsumexp(tf.concat(sample_log_py, 1), axis=1)
            all_batch_size = tf.shape(all_batch_angles)[0]
            mean_inputs = tf.tile(adaptation_params['mu'], [all_batch_size, 1])
            mean__images = generate_views(all_batch_angles, mean_inputs)
            return [task_log_py, mean__images]

        # tf mapping of batch to evaluation function
        batch_output = tf.map_fn(fn=evaluate_task,
                                 elems=(batch_train_images, batch_train_angles, batch_test_images, batch_test_angles,
                                        batch_all_angles),
                                 dtype=[tf.float32, tf.float32],
                                 parallel_iterations=test_tasks_per_batch)

        batch_log_densities, generated_images = batch_output

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, save_path=args.model_path)
            train_inputs, test_inputs, train_outputs, test_outputs = \
                data.get_batch('test', test_tasks_per_batch, test_shot)
            feed_dict = {batch_train_images: train_inputs, batch_test_images: test_inputs,
                         batch_train_angles: train_outputs, batch_test_angles: test_outputs,
                         batch_all_angles: all_angles}
            images_log_density, test_generated_images = sess.run([batch_log_densities, generated_images], feed_dict)
            # Bits-per-pixel computations
            images_log_density = np.array(images_log_density)
            images_bits = -images_log_density / math.log(math.e, 2)
            bits_per_pixel = images_bits / (data.get_image_height() * data.get_image_width())
            task_bits_per_pixel.append(bits_per_pixel)
            all_gt_images = np.concatenate((train_inputs, test_inputs), axis=1)
            all_gt_angles = np.concatenate((train_outputs, test_outputs), axis=1)
            quality_metrics = compute_image_quality_metrics(all_gt_images[0], all_gt_angles[0],
                                                            test_generated_images[0], all_angles[0])
            task_mse.append(quality_metrics[0])
            task_nrmse.append(quality_metrics[1])
            task_ssim.append(quality_metrics[2])
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            output_path = os.path.join(args.model_path, 'output_composite_{0:02d}'.format(test_iteration))
            plot_image_strips(train_inputs[0], test_generated_images[0], all_angles[0], all_gt_images[0],
                              all_gt_angles[0], data.get_image_height(), data.get_image_width(), angles_to_plot,
                              output_path)
            output_folder = os.path.join(args.model_path, 'images_{0:02d}'.format(test_iteration))
            save_images_to_folder(test_generated_images[0], all_angles[0], all_gt_images[0], all_gt_angles[0],
                                  output_folder)

        tf.reset_default_graph()
        test_iteration += 1

    model_bpp = np.array(task_bits_per_pixel).mean()
    model_bpp_err = (1.96 * np.array(task_bits_per_pixel).std()) / np.sqrt(len(task_bits_per_pixel))
    print('Model average BPP: {0:5.5f} +/- {1:5.5f}'.format(model_bpp, model_bpp_err))

    model_mse = np.array(task_mse).mean()
    model_mse_err = (1.96 * np.array(task_mse).std()) / np.sqrt(len(task_mse))
    print('Model average MSE: {0:5.5f} +/- {1:5.5f}'.format(model_mse, model_mse_err))

    model_nrmse = np.array(task_nrmse).mean()
    model_nrmse_err = (1.96 * np.array(task_nrmse).std()) / np.sqrt(len(task_nrmse))
    print('Model average NRMSE: {0:5.5f} +/- {1:5.5f}'.format(model_nrmse, model_nrmse_err))

    model_ssim = np.array(task_ssim).mean()
    model_ssim_err = (1.96 * np.array(task_ssim).std()) / np.sqrt(len(task_ssim))
    print('Model average SSIM: {0:5.5f} +/- {1:5.5f}'.format(model_ssim, model_ssim_err))

    print('Output images saved in: {0:}'.format(args.model_path))


if __name__ == "__main__":
    tf.app.run()
