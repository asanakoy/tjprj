# Original work Copyright 2015 The TensorFlow Authors. Licensed under the Apache License v2.0 http://www.apache.org/licenses/LICENSE-2.0
# Modified work Copyright (c) 2016 Artsiom Sanakoyeu

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
import pprint
from tfext import network_spec
import tfext.alexnet
import tfext.utils
from trainhelper import trainhelper
import batch_loader_with_prefetch
import matplotlib.pyplot as plt
import gc


def get_pathes(category, dataset, is_bbox_sq):
    if is_bbox_sq:
        data_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/crops/{}/images_227x227_bbox_sq.mat'.format(dataset, category)
    else:
        data_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/crops/{}/images.mat'.format(dataset, category)

    mean_path = os.path.join('/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' + category + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/mean.npy')
    output_dir = os.path.join(os.path.expanduser('~/tmp/tf_test'))
    return data_path, mean_path, output_dir


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


def setup_network(**params):

    # If a network exists clear all ops, variables and tensors before creating a new instance
    if 'net' in params.keys():
        params['net'].sess.close()
        tf.reset_default_graph()
        del params['net']
        del params['loss']
        del params['train_op']

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
    return {'net': net, 'train_op': train_op, 'loss': loss, 'saver': saver, 'summary_writer': summary_writer, 'summary': summary}


def run_training_current_clustering(**params):

    # plotter = Plotter(2, 2)
    log_step = 1
    summary_step = 200
    for step in xrange(params['max_iter']):

        start_time = time.time()

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        feed_dict = tfext.utils.fill_feed_dict(params['net'], params['batch_ldr'],
                                               batch_size=params['batch_size'],
                                               phase='train')

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        if step % summary_step == 0:
            summary_str, _, loss_value = params['net'].sess.run([params['summary'], params['train_op'], params['loss']],
                                                      feed_dict=feed_dict)
            params['summary_writer'].add_summary(summary_str, step)
            # summary_writer.flush()
        else:
            _, loss_value = params['net'].sess.run([params['train_op'], params['loss']], feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % log_step == 0 or step + 1 == params['max_iter']:
            print('Step %d: loss = %.2f (%.3f s, %.2f im/s)'
                  % (step, loss_value, duration,
                     params['batch_size'] / duration))
    return params


def run_training(**params):

    params_clustering = trainhelper.get_params_clustering(params['dataset'], params['category'])

    for clustering_round in range(0, 5):

        params['clustering_round'] = 'clustering_round'
        # Delete old batch_ldr, recompute clustering and create new batch_ldr
        if 'batch_ldr' in params:
            params['batch_ldr']
            gc.collect()

        # Use HOGLDA for initial estimate of similarities
        if clustering_round == 0:
            matrices = trainhelper.get_step_similarities(0, None, params['category'], None,
                                                         pathtosim=params_clustering['pathtosim'],
                                                         pathtosim_avg=params_clustering['pathtosim_avg'])
        else:
            matrices = trainhelper.get_step_similarities(clustering_round, params['net'], params['category'], ['fc7'])

        # Run clustering and update corresponding param fields
        params_clustering.update(matrices)
        params_clustering['clustering_round'] = clustering_round
        batch_ldr_dict_params, params_clustering = trainhelper.runClustering(**params_clustering)
        params['indexfile_path'] = batch_ldr_dict_params
        params['num_classes'] = batch_ldr_dict_params['labels'].max() + 1
        params['batch_ldr'] = batch_loader_with_prefetch.BatchLoaderWithPrefetch(params)

        # Create network with new clustering parameters and return it in network_params dict
        network_params = setup_network(**params)
        params.update(network_params)

        # Restore from previous round model
        if clustering_round > 0:
            checkpoint_file_round = checkpoint_file + '-' + str(clustering_round)
            params['net'].restore_from_snapshot(checkpoint_file_round, 7)

        # Run training and save snapshot
        params = run_training_current_clustering(**params)
        checkpoint_file = os.path.join(params['output_dir'], 'checkpoint')
        params['saver'].save(params['net'].sess, checkpoint_file, global_step=clustering_round + 1)

    params['net'].sess.close()


def main(argv):
    if len(argv) == 0:
        argv = ['0']
    if len(argv) > 1:
        category = argv[1]
    else:
        category = 'long_jump'
    dataset = 'OlympicSports'

    # should we use crops that were cropped using the biggest square bounding box around the person
    is_bbox_sq = 1

    data_path, mean_path, output_dir = get_pathes(category, dataset, is_bbox_sq)

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
        'num_classes': None,
        'snapshot_iter': 2000,
        'max_iter': 10000,
        'indexing_1_based': 0,
        'images_mat_filepath': data_path,
        'indexfile_path': None,
        'mean_filepath': mean_path,
        'seed': 1988,
        'output_dir': output_dir,
        'init_model': get_first_model_path(dataset),
        'device_id': '/gpu:{}'.format(int(argv[0])),
        'shuffle_every_epoch': False,
        'online_augmentations': True,
        'async_preload': True,
        'num_data_workers': 5,
        'gpu_memory_fraction': 0.4,
        'augmenter_params': dict(hflip=True, vflip=False,
                                 scale_to_percent=(0.7, 1.3),
                                 scale_axis_equally=True,
                                 rotation_deg=10, shear_deg=7,
                                 translation_x_px=30, translation_y_px=30)
    }
    with open(os.path.join(output_dir, 'train_params.dump.txt'), 'w') as f:
        f.write('{}\n'.format(pprint.pformat(params)))
    run_training(**params)


if __name__ == '__main__':
    main(sys.argv[1:])
