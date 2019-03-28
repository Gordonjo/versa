"""
Script to reproduce the few-shot classification results in:
"Meta-Learning Probabilistic Inference For Prediction"
https://arxiv.org/pdf/1805.09921.pdf

The following command lines will reproduce the published results within error-bars:

Omniglot 5-way, 5-shot
----------------------
python run_classifier.py

Omniglot 5-way, 1-shot
----------------------
python run_classifier.py --shot 1

Omniglot 20-way, 5-shot
-----------------------
python run_classifier.py --way 20 --iterations 60000

Omniglot 20-way, 1-shot
-----------------------
python run_classifier.py --way 20 --shot 1 --iterations 100000

minImageNet 5-way, 5-shot
-------------------------
python run_classifier.py --dataset miniImageNet --tasks_per_batch 4 --iterations 100000 --dropout 0.5

minImageNet 5-way, 1-shot
-------------------------
python run_classifier.py --dataset miniImageNet --shot 1 --tasks_per_batch 8 --iterations 50000 --dropout 0.5 -lr 0.00025

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
from features import extract_features_omniglot, extract_features_mini_imagenet
from inference import infer_classifier
from utilities import sample_normal, multinoulli_log_density, print_and_log, get_log_files
from data import get_data

"""
parse_command_line: command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["Omniglot", "miniImageNet"],
                        default="Omniglot", help="Dataset to use")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run traing only, testing only, or both training and testing.")
    parser.add_argument("--d_theta", type=int, default=256,
                        help="Size of the feature extractor output.")
    parser.add_argument("--shot", type=int, default=5,
                        help="Number of training examples.")
    parser.add_argument("--way", type=int, default=5,
                        help="Number of classes.")
    parser.add_argument("--test_shot", type=int, default=None,
                        help="Shot to be used at evaluation time. If not specified 'shot' will be used.")
    parser.add_argument("--test_way", type=int, default=None,
                        help="Way to be used at evaluation time. If not specified 'way' will be used.")
    parser.add_argument("--tasks_per_batch", type=int, default=16,
                        help="Number of tasks per batch.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples from q.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--iterations", type=int, default=80000,
                        help="Number of training iterations.")
    parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint',
                        help="Directory to save trained models.")
    parser.add_argument("--dropout", type=float, default=0.9,
                        help="Dropout keep probability.")
    parser.add_argument("--test_model_path", "-m", default=None,
                        help="Model to load and test.")
    parser.add_argument("--print_freq", type=int, default=200, 
                        help="Frequency of summary results (in iterations).")
    args = parser.parse_args()

    # adjust test_shot and test_way if necessary
    if args.test_shot is None:
        args.test_shot = args.shot
    if args.test_way is None:
        args.test_way = args.way

    return args

    
def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = parse_command_line()

    logfile, checkpoint_path_validation, checkpoint_path_final = get_log_files(args.checkpoint_dir)

    print_and_log(logfile, "Options: %s\n" % args)

    # Load training and eval data
    data = get_data(args.dataset)

    # set the feature extractor based on the dataset
    feature_extractor_fn = extract_features_mini_imagenet
    if args.dataset == "Omniglot":
        feature_extractor_fn = extract_features_omniglot

    # evaluation samples
    eval_samples_train = 15
    eval_samples_test = args.shot

    # testing parameters
    test_iterations = 600
    test_args_per_batch = 1  # always use a batch size of 1 for testing

    # tf placeholders
    train_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               data.get_image_height(),
                                               data.get_image_width(),
                                               data.get_image_channels()],
                                  name='train_images')
    test_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              data.get_image_height(),
                                              data.get_image_width(),
                                              data.get_image_channels()],
                                 name='test_images')
    train_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               args.way],
                                  name='train_labels')
    test_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              args.way],
                                 name='test_labels')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    L = tf.constant(args.samples, dtype=tf.float32, name="num_samples")

    # Relevant computations for a single task
    def evaluate_task(inputs):
        train_inputs, train_outputs, test_inputs, test_outputs = inputs
        with tf.variable_scope('shared_features'):
            # extract features from train and test data
            features_train = feature_extractor_fn(images=train_inputs,
                                                  output_size=args.d_theta,
                                                  use_batch_norm=True,
                                                  dropout_keep_prob=dropout_keep_prob)
            features_test = feature_extractor_fn(images=test_inputs,
                                                 output_size=args.d_theta,
                                                 use_batch_norm=True,
                                                 dropout_keep_prob=dropout_keep_prob)
        # Infer classification layer from q
        with tf.variable_scope('classifier'):
            classifier = infer_classifier(features_train, train_outputs, args.d_theta, args.way)

        # Local reparameterization trick
        # Compute parameters of q distribution over logits
        weight_mean, bias_mean = classifier['weight_mean'], classifier['bias_mean']
        weight_log_variance, bias_log_variance = classifier['weight_log_variance'], classifier['bias_log_variance']
        logits_mean_test = tf.matmul(features_test, weight_mean) + bias_mean
        logits_log_var_test =\
            tf.log(tf.matmul(features_test ** 2, tf.exp(weight_log_variance)) + tf.exp(bias_log_variance))
        logits_sample_test = sample_normal(logits_mean_test, logits_log_var_test, args.samples)
        test_labels_tiled = tf.tile(tf.expand_dims(test_outputs, 0), [args.samples, 1, 1])
        task_log_py = multinoulli_log_density(inputs=test_labels_tiled, logits=logits_sample_test)
        averaged_predictions = tf.reduce_logsumexp(logits_sample_test, axis=0) - tf.log(L)
        task_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_outputs, axis=-1),
                                                        tf.argmax(averaged_predictions, axis=-1)), tf.float32))
        task_score = tf.reduce_logsumexp(task_log_py, axis=0) - tf.log(L)
        task_loss = -tf.reduce_mean(task_score, axis=0)

        return [task_loss, task_accuracy]

    # tf mapping of batch to evaluation function
    batch_output = tf.map_fn(fn=evaluate_task,
                             elems=(train_images, train_labels, test_images, test_labels),
                             dtype=[tf.float32, tf.float32],
                             parallel_iterations=args.tasks_per_batch)

    # average all values across batch
    batch_losses, batch_accuracies = batch_output
    loss = tf.reduce_mean(batch_losses)
    accuracy = tf.reduce_mean(batch_accuracies)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if args.mode == 'train' or args.mode == 'train_test':
            # train the model
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_step = optimizer.minimize(loss)
    
            validation_batches = 200
            iteration = 0
            best_validation_accuracy = 0.0
            train_iteration_accuracy = []
            sess.run(tf.global_variables_initializer())
            # Main training loop
            while iteration < args.iterations:
                train_inputs, test_inputs, train_outputs, test_outputs = \
                    data.get_batch('train', args.tasks_per_batch, args.shot, args.way, eval_samples_train)

                feed_dict = {train_images: train_inputs, test_images: test_inputs,
                             train_labels: train_outputs, test_labels: test_outputs,
                             dropout_keep_prob: args.dropout}
                _, iteration_loss, iteration_accuracy = sess.run([train_step, loss, accuracy], feed_dict)
                train_iteration_accuracy.append(iteration_accuracy)
                if (iteration > 0) and (iteration % args.print_freq == 0):
                    # compute accuracy on validation set
                    validation_iteration_accuracy = []
                    validation_iteration = 0
                    while validation_iteration < validation_batches:
                        train_inputs, test_inputs, train_outputs, test_outputs = \
                            data.get_batch('validation', args.tasks_per_batch, args.shot, args.way, eval_samples_test)
                        feed_dict = {train_images: train_inputs, test_images: test_inputs,
                                     train_labels: train_outputs, test_labels: test_outputs,
                                     dropout_keep_prob: 1.0}
                        iteration_accuracy = sess.run(accuracy, feed_dict)
                        validation_iteration_accuracy.append(iteration_accuracy)
                        validation_iteration += 1
                    validation_accuracy = np.array(validation_iteration_accuracy).mean()
                    train_accuracy = np.array(train_iteration_accuracy).mean()

                    # save checkpoint if validation is the best so far
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        saver.save(sess=sess, save_path=checkpoint_path_validation)

                    print_and_log(logfile, 'Iteration: {}, Loss: {:5.3f}, Train-Acc: {:5.3f}, Val-Acc: {:5.3f}'
                        .format(iteration, iteration_loss, train_accuracy, validation_accuracy))
                    train_iteration_accuracy = []

                iteration += 1
            # save the checkpoint from the final epoch
            saver.save(sess, save_path=checkpoint_path_final)
            print_and_log(logfile, 'Fully-trained model saved to: {}'.format(checkpoint_path_final))
            print_and_log(logfile, 'Best validation accuracy: {:5.3f}'.format(best_validation_accuracy))
            print_and_log(logfile, 'Best validation model saved to: {}'.format(checkpoint_path_validation))
        
        def test_model(model_path, load=True):
            if load:
                saver.restore(sess, save_path=model_path)
            test_iteration = 0
            test_iteration_accuracy = []
            while test_iteration < test_iterations:
                train_inputs, test_inputs, train_outputs, test_outputs = \
                                        data.get_batch('test', test_args_per_batch, args.test_shot, args.test_way,
                                                       eval_samples_test)
                feedDict = {train_images: train_inputs, test_images: test_inputs,
                            train_labels: train_outputs, test_labels: test_outputs,
                            dropout_keep_prob: 1.0}
                iter_acc = sess.run(accuracy, feedDict)
                test_iteration_accuracy.append(iter_acc)
                test_iteration += 1
            test_accuracy = np.array(test_iteration_accuracy).mean() * 100.0
            confidence_interval_95 = \
                (196.0 * np.array(test_iteration_accuracy).std()) / np.sqrt(len(test_iteration_accuracy))
            print_and_log(logfile, 'Held out accuracy: {0:5.3f} +/- {1:5.3f} on {2:}'
                          .format(test_accuracy, confidence_interval_95, model_path))

        if args.mode == 'train_test':
            print_and_log(logfile, 'Train Shot: {0:d}, Train Way: {1:d}, Test Shot {2:d}, Test Way {3:d}'
                          .format(args.shot, args.way, args.test_shot, args.test_way))
            # test the model on the final trained model
            # no need to load the model, it was just trained
            test_model(checkpoint_path_final, load=False)

            # test the model on the best validation checkpoint so far
            test_model(checkpoint_path_validation)

        if args.mode == 'test':
            test_model(args.test_model_path)

    logfile.close()


if __name__ == "__main__":
    tf.app.run()
