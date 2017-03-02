# Copyright (c) 2016 Artsiom Sanakoyeu

# pylint: disable=missing-docstring
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys

import tensorflow as tf
import h5py
import numpy as np
from tfext import network_spec
import tfext.alexnet
import tfext.utils
import batch_loader
import matplotlib.pyplot as plt


def get_pathes(category, dataset):
    data_path = os.path.join('/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/' + dataset + '/augmented_data/10T/training_data_' + dataset + '_')
    indices_dir = os.path.join('/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' + category + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/')
    # output_dir = os.pathjoin('/export/home/mbautist/Desktop/workspace/cnn_similarities/ablation_experiments/outputs/snapshots/', category)
    output_dir = os.path.join(os.path.expanduser('~/tmp/tf_test'))
    return data_path, indices_dir, output_dir


def get_num_classes(indices_path):
    mat_data = h5py.File(indices_path, 'r')
    if 'labels' in mat_data.keys():
        num_cliques = int(np.array(mat_data['labels']).max() + 1)
    else:
        num_cliques = int(np.array(mat_data['new_labels']).max() + 1)
    return num_cliques


def get_first_model_path(dataset):
    if dataset == 'OlympicSports':
        return '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.npy'
    elif dataset == 'VOC':
        raise NotImplementedError
        return None


class Plotter:
    def __init__(self, r, c):
        plt.ion()
        self.f, self.axes = plt.subplots(r, c)
        self.styles = ['bo-', 'ro-'] * (r*c)
        self.axes = self.axes.reshape(-1)

    def plot(self, it, data):
        for i, (key, val) in enumerate(data.iteritems()):
            self.axes[i].plot(it, val, self.styles[i], label=key)
            if iter == 0:
                self.axes[i].set_xlabel('Iteration')
                self.axes[i].legend()
        plt.pause(0.0000001)


def run_training(**params):
    with tf.Graph().as_default():
        net = tfext.alexnet.Alexnet(**params)
        logits = net.fc8
        loss = network_spec.loss(logits, net.y_gt)
        train_op = network_spec.training(net, loss, **params)

        # Add the Op to compare the logits to the labels during correct_classified_top1.
        eval_correct_top1 = network_spec.correct_classified_top1(logits, net.y_gt)
        accuracy = tf.cast(eval_correct_top1, tf.float32) / \
                   tf.constant(params['batch_size'], dtype=tf.float32)

        saver = tf.train.Saver()

        # Instantiate a SummaryWriter to output summaries and the Graph of the current sesion.
        summary_writer = tf.train.SummaryWriter(params['output_dir'], net.sess.graph)
        summary = tf.scalar_summary(['loss', 'batch_accuracy'], [loss, accuracy])

        net.sess.run(tf.initialize_all_variables())

        batch_ldr = batch_loader.BatchLoader(params)

        # plotter = Plotter(2, 2)
        log_step = 1
        summary_step = 200
        for step in xrange(params['max_iter']):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = tfext.utils.fill_feed_dict(net, batch_ldr,
                                                   batch_size=params['batch_size'],
                                                   phase='train')

            if step % summary_step == 0:
                global_step, summary_str, _, loss_value = net.sess.run([net.global_iter_counter, summary, train_op, loss],
                                                          feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=global_step)
                # summary_writer.flush()
            else:
                global_step, _, loss_value = net.sess.run([net.global_iter_counter, train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step != 0 and \
                    (step + 1 % params['snapshot_iter'] == 0 or step + 1 == params['max_iter']):
                checkpoint_file = os.path.join(params['output_dir'], 'checkpoint')
                saver.save(net.sess, checkpoint_file, global_step=global_step)

            duration_full = time.time() - start_time
            if step % log_step == 0 or step + 1 == params['max_iter']:
                print('Step %d: loss = %.2f (%.3f s, %.2f im/s) (full: %.3f s)'
                      % (step, loss_value, duration,
                         params['batch_size'] / duration, duration_full))
    net.sess.close()


def main(argv):
    if len(argv) == 0:
        argv = ['0']
    if len(argv) > 1:
        category = argv[1]
    else:
        category = 'vault'
    dataset = 'OlympicSports'

    data_path, indices_dir, output_dir = get_pathes(category, dataset)
    images_aug_path = os.path.join(data_path + category + '.mat')
    train_indices_path = os.path.join(indices_dir,
                                      category + '_batch_128_10trans_shuffleMB1shuffleALL_0_train.mat')
    mean_path = os.path.join(indices_dir, 'mean.npy')
    num_cliques = get_num_classes(train_indices_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = {
        'im_shape': (227, 227, 3),
        'batch_size': 128,
        'base_lr': 0.001,
        'fc_lr_mult': 1.0,
        'conv_lr_mult': 0.1,
        'num_layers_to_init': 6,
        'dataset': dataset,
        'category': category,
        'num_classes': num_cliques,
        'snapshot_iter': 2000,
        'max_iter': 20000,
        'indexing_1_based': 1,
        'images_mat_filepath': images_aug_path,
        'indexfile_path': train_indices_path,
        'mean_filepath': mean_path,
        'seed': 1988,
        'output_dir': output_dir,
        'init_model': get_first_model_path(dataset),
        'gpu_memory_fraction': 0.38,
        'device_id': '/gpu:{}'.format(int(argv[0]))
    }
    run_training(**params)


if __name__ == '__main__':
    main(sys.argv[1:])
