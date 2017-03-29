# Copyright (c) 2016 Artsiom Sanakoyeu

# pylint: disable=missing-docstring
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import tensorflow as tf
import h5py
import numpy as np
from tfext import network_spec
import tfext.alexnet
import tfext.utils
import tfext.centroider
import batch_loader
import sys

import matplotlib.pyplot as plt
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('gpu', '0', 'Gpu id to use')


def get_pathes(category, dataset, ft):
    data_path = os.path.join('/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/' + dataset + '/augmented_data/10T/training_data_' + dataset + '_')
    if category == 'long_jump':
        indices_dir = os.path.join('/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' + category + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/')
    else:
        indices_dir = os.path.join(
            '/export/home/mbautist/Desktop/workspace/cnn_similarities/MIL-CliqueCNN/clustering/LSP/iter_1')
    output_dir = os.path.join(os.path.expanduser('~/tmp/tf_test_' + category + '_ftCliqueCNN_' + ft + '/'))
    return data_path, indices_dir, output_dir


def get_num_classes(category, indices_path):
    mat_data = h5py.File(indices_path, 'r')
    # num_cliques = int(np.array(mat_data['new_labels']).max() + 1)
    if category == 'long_jump':
        num_cliques = int(np.array(mat_data['new_labels']).max() + 1)
    else:
        num_cliques = int(np.array(mat_data['labels']).max() + 1)
    return num_cliques


def get_first_model_path(dataset):
    if dataset == 'OlympicSports':
        return '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.npy'
    elif dataset == 'VOC':
        # we need to convert WangVideoTriplet/color_model.caffemodel
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
        norm_fc7 = tf.nn.l2_normalize(net.fc7, dim=1)
        mu_pl = tf.placeholder(tf.float32, (None, 4096))
        unique_mu_pl = tf.placeholder(tf.float32, (None, 4096))
        sigma_pl = tf.placeholder(tf.float32, (1,))
        loss = network_spec.loss_magnet(norm_fc7, mu_pl, unique_mu_pl, sigma_pl, net.y_gt)
        train_op = network_spec.training(net, loss, **params)


        # Add the Op to compare the logits to the labels during correct_classified_top1.
        # eval_correct_top1 = network_spec.correct_classified_top1(logits, net.y_gt)
        summary = tf.summary.merge_all()
        saver = tf.train.Saver()

        # Instantiate a SummaryWriter to output summaries and the Graph of the current sesion.
        summary_writer = tf.summary.FileWriter(params['output_dir'], net.sess.graph)
        net.sess.run(tf.global_variables_initializer())

        if params['ftCliqueCNN']:
            snapshot_path = '/export/home/asanakoy/workspace01/datasets/OlympicSports/cnn/long_jump/checkpoint-20000'
            net.restore_from_snapshot(snapshot_path, 7)


        batch_ldr = batch_loader.BatchLoader(params)
        centroid = tfext.centroider.Centroider(batch_ldr)
        centroid.updateCentroids(net.sess, net.x, net.fc7)


        # plotter = Plotter(2, 2)

        log_step = 1
        for step in xrange(params['max_iter']):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = tfext.utils.fill_feed_dict_magnet(net, mu_pl, unique_mu_pl, sigma_pl, batch_ldr, centroid,
                                                   batch_size=params['batch_size'],
                                                   phase='train')

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = net.sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 5000 == 0 and step != 0:
                centroid.updateCentroids(net.sess, net.x, net.fc7)
            # Write the summaries and print an overview fairly often.
            if step % log_step == 0:
                # data = OrderedDict()
                # data['loss'] = loss_value
                # plotter.plot(step, data)
                # Update the events file.
                summary_str = net.sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % params['snapshot_iter'] == 0 or (step + 1) == params['max_iter']:
                checkpoint_file = os.path.join(params['output_dir'], 'checkpoint')
                saver.save(net.sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                # print('Training Data Eval:')
                # tfext.utils.calc_acuracy(net, net.sess,
                #                          eval_correct_top1,
                #                          batch_ldr,
                #                          batch_size=params['batch_size'],
                #                          num_images=1000)

            duration_full = time.time() - start_time
            if step % log_step == 0:
                print('Step %d: loss = %.2f (%.3f s) (full: %.3f s)' % (step, loss_value,
                                                                        duration, duration_full))


def main(category, finetune):
    if category == 'long_jump':
        dataset = 'OlympicSports'
    else:
        dataset = 'lsp_dataset_original'



    data_path, indices_dir, output_dir = get_pathes(category, dataset, finetune)
    images_aug_path = os.path.join(data_path + category + '.mat')
    if category == 'long_jump':
        train_indices_path = os.path.join(indices_dir, category + '_batch_128_10trans_shuffleMB1shuffleALL_0_train.mat')
    else:
        train_indices_path = os.path.join(indices_dir + '/train_indices.mat')

    mean_path = os.path.join(indices_dir, 'mean.npy')
    num_cliques = get_num_classes(category, train_indices_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    params = {
        'im_shape': (227, 227, 3),
        'batch_size': 128,
        'base_lr': 0.001,
        'fc_lr_mult': 1.0,
        'conv_lr_mult': 1.0,
        'num_layers_to_init': 7,
        'dataset': dataset,
        'category': category,
        'num_classes': num_cliques,
        'snapshot_iter': 5000,
        'max_iter': 20000,
        'ftCliqueCNN': bool(finetune),
        'random_init_type': tfext.alexnet.Alexnet.RandomInitType.GAUSSIAN,
        'indexing_1_based': 1,
        'images_mat_filepath': images_aug_path,
        'indexfile_path': train_indices_path,
        'mean_filepath': mean_path,
        'shuffle_every_epoch': False,
        'seed': 1988,
        'output_dir': output_dir,
        'init_model': get_first_model_path(dataset),
        'device_id': '/gpu:{}'.format(FLAGS.gpu)
    }


    run_training(**params)


if __name__ == '__main__':
    category = sys.argv[1]
    finetune = sys.argv[2]
    main(category, finetune)
