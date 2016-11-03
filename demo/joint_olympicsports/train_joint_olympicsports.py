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
import gc
import pprint
import matplotlib.pyplot as plt
from tfext import network_spec
import tfext.alexnet
import tfext.utils
from trainhelper import trainhelper
import batch_loader_with_prefetch
import eval.olympicsports.roc.roc_from_net
import eval.olympicsports.utils
from helper import CATEGORIES
from helper import BatchManager


def get_pathes(is_bbox_sq):
    assert not is_bbox_sq
    images_mat_pathes = {cat: '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/{}/images.mat'.format(
                        cat) for cat in CATEGORIES}

    output_dir = os.path.expanduser('/export/home/asanakoy/workspace/OlympicSports/cnn/joint_categories_0.1conv_anchors')
    # output_dir = os.path.join(os.path.expanduser('~/tmp/tf_test'))
    mean_path = os.path.join(output_dir, 'mean.npy')
    return images_mat_pathes, mean_path, output_dir


def get_num_classes(indices_path):
    # TODO:
    mat_data = h5py.File(indices_path, 'r')
    if 'labels' in mat_data.keys():
        num_cliques = int(np.array(mat_data['labels']).max() + 1)
    else:
        num_cliques = int(np.array(mat_data['new_labels']).max() + 1)
    return num_cliques


def get_first_model_path():
    return '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.npy'


def setup_network(num_classes, **params):
    net = tfext.alexnet.Alexnet(num_classes=num_classes, **params)
    logits = net.fc8
    loss_op = network_spec.loss(logits, net.y_gt)
    train_op = network_spec.training(net, loss_op, **params)

    # Add the Op to compare the logits to the labels during correct_classified_top1.
    eval_correct_top1 = network_spec.correct_classified_top1(logits, net.y_gt)
    accuracy = tf.cast(eval_correct_top1, tf.float32) / \
               tf.constant(params['batch_size'], dtype=tf.float32)

    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph of the current sesion.
    summary_writer = tf.train.SummaryWriter(params['output_dir'], net.sess.graph)
    summary_op = tf.scalar_summary(['loss', 'batch_accuracy'], [loss_op, accuracy])

    net.sess.run(tf.initialize_all_variables())
    return net, train_op, loss_op, saver, summary_writer, summary_op


def run_training_current_clustering(net, batch_manager, train_op, loss_op, summary_op, summary_writer,
                                    saver, **params):
    log_step = 1
    summary_step = 5
    for step in xrange(params['max_iter']):

        start_time = time.time()
        feed_dict = batch_manager.fill_feed_dict(phase='train')

        if step % summary_step == 0:
            global_step, summary_str, _, loss_value = net.sess.run(
                [net.global_iter_counter,
                 summary_op,
                 train_op,
                 loss_op],
                feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step)
        else:
            global_step, _, loss_value = net.sess.run(
                [net.global_iter_counter, train_op, loss_op],
                feed_dict=feed_dict)

        if step > 1 and (step % params['test_step'] == 0 or step + 1 == params['max_iter']):
            aucs = list()
            for category in CATEGORIES:
                roc_auc = eval.olympicsports.roc. \
                    roc_from_net.compute_roc_auc_from_net(net,
                                                          category,
                                                          ['fc7'],
                                                          mat_path=params['images_mat_pathes'][category],
                                                          mean_path=params['mean_filepath'],
                                                          batch_size=256,
                                                          norm_method=None)
                aucs.append(roc_auc)
                summary_writer.add_summary(tfext.utils.create_sumamry('ROCAUC_{}'.format(category), roc_auc),
                                                                      global_step=global_step)
                print('Step %d: %s ROCAUC = %.2f' % (step, category, roc_auc))
            summary_writer.add_summary(tfext.utils.create_sumamry('mROCAUC', np.mean(aucs)),
                                       global_step=global_step)
            summary_writer.flush()
        duration = time.time() - start_time

        if step % params['snapshot_iter'] == 0:
            # TODO: write the number of round in the name
            checkpoint_prefix = os.path.join(params['output_dir'], 'checkpoint')
            saver.save(net.sess, checkpoint_prefix, global_step=global_step)

        if step % log_step == 0 or step + 1 == params['max_iter']:
            print('Step %d: loss = %.2f (%.3f s, %.2f im/s)'
                  % (step, loss_value, duration,
                     params['batch_size'] / duration))


def cluster_category(clustering_round, category, net, output_dir, params_clustering):
    # Delete old batch_ldr, recompute clustering and create new batch_ldr
    # Use HOGLDA for initial estimate of similarities
    print('Clustering %s' % category)
    if clustering_round == 0:
        matrices = trainhelper.get_step_similarities(step=0,
                                                     net=None,
                                                     category=category,
                                                     dataset='OlympicSports',
                                                     layers=None,
                                                     pathtosim=params_clustering['pathtosim'],
                                                     pathtosim_avg=params_clustering['pathtosim_avg'])
    else:
        matrices = trainhelper.get_step_similarities(clustering_round,
                                                     net=net,
                                                     category=category,
                                                     dataset='OlympicSports',
                                                     layers=['fc7'])

    # Run clustering and update corresponding param fields
    params_clustering.update(matrices)
    params_clustering['clustering_round'] = clustering_round
    params_clustering['output_dir'] = output_dir
    index_dict, _ = trainhelper.runClustering(**params_clustering)
    return index_dict


def run_training(**params):

    net = None
    checkpoint_prefix = os.path.join(params['output_dir'], 'round_checkpoint')
    eval.olympicsports.utils.get_joint_categories_mean(params['mean_filepath'])
    assert os.path.exists(params['mean_filepath'])

    for clustering_round in range(0, params['num_clustering_rounds']):

        num_classes = 0
        batch_loaders = dict()
        for cat in CATEGORIES:
            params_clustering = trainhelper.get_params_clustering('OlympicSports', cat)
            # set num batches of cliques to number of anchors
            params_clustering['init_nbatches'] = len(params_clustering['anchors']['anchor'])
            index_dict = cluster_category(clustering_round,
                                          cat,
                                          net,
                                          params['output_dir'],
                                          params_clustering)
            index_dict['labels'] = np.asarray(index_dict['labels']) + num_classes
            num_classes = index_dict['labels'].max() + 1

            batch_loader_params = params.copy()
            batch_loader_params['indexing_1_based'] = 0
            batch_loader_params['indexfile_path'] = index_dict
            batch_loader_params['images_mat_filepath'] = params['images_mat_pathes'][cat]
            batch_loaders[cat] = batch_loader_with_prefetch.BatchLoaderWithPrefetch(batch_loader_params)

        batch_manager = BatchManager(batch_loaders,
                                     params['batch_size'],
                                     params['im_shape'][:2],
                                     random_shuffle=params['random_shuffle_categories'],
                                     random_seed=params['seed'])
        # Create network with new clustering parameters
        # If a network exists close session and reset graph
        if net is not None:
            net.sess.close()
        tf.reset_default_graph()
        net, train_op, loss_op, saver, summary_writer, summary_op = setup_network(num_classes, **params)
        # Restore from previous round model
        if clustering_round > 0:
            checkpoint_path = checkpoint_prefix + '-' + str(clustering_round)
            net.restore_from_snapshot(checkpoint_path, 7, restore_iter_counter=True)

        # Run training and save snapshot
        run_training_current_clustering(net, batch_manager, train_op, loss_op, summary_op, summary_writer,
                                        saver, **params)
        saver.save(net.sess, checkpoint_prefix,
                             global_step=clustering_round + 1)
        batch_manager.cleanup()
        gc.collect()

    if net is not None:
        net.sess.close()


def main(argv):
    images_mat_pathes, mean_path, output_dir = get_pathes(is_bbox_sq=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = {
        'im_shape': (227, 227, 3),
        'batch_size': 128,
        'base_lr': 0.001,
        'fc_lr_mult': 1.0,
        'conv_lr_mult': 0.1,
        'num_layers_to_init': 6,
        'dataset': 'OlympicSports',
        'images_mat_pathes': images_mat_pathes,
        'max_iter': 2000000,
        'snapshot_iter': 10000,
        'test_step': 10000,
        'num_clustering_rounds': 1,
        'random_shuffle_categories': True,
        'mean_filepath': mean_path,
        'seed': 1988,
        'output_dir': output_dir,
        'init_model': get_first_model_path(),
        'device_id': '/gpu:{}'.format(0),
        'shuffle_every_epoch': True,
        'online_augmentations': False,
        'async_preload': False,
        'num_data_workers': 3,
        'gpu_memory_fraction': None,
        'augmenter_params': dict(hflip=False, vflip=False,
                                 scale_to_percent=(1.0 / 2 ** 0.5, 2 ** 0.5),
                                 scale_axis_equally=True,
                                 rotation_deg=5, shear_deg=4,
                                 translation_x_px=15, translation_y_px=15)
    }
    with open(os.path.join(output_dir, 'train_params.dump.txt'), 'w') as f:
        f.write('{}\n'.format(pprint.pformat(params)))
    run_training(**params)


if __name__ == '__main__':
    main(sys.argv[1:])
