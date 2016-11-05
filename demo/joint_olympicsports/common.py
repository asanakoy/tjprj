from __future__ import division

import os.path
import time
import sys
import tensorflow as tf
import h5py
import numpy as np
import gc
from tfext import network_spec
import tfext.alexnet
import tfext.convnet
import tfext.utils
from trainhelper import trainhelper
import batch_loader_with_prefetch
import eval.olympicsports.roc.roc_from_net
import eval.olympicsports.utils
from helper import CATEGORIES
from helper import BatchManager


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


def setup_alexnet_network(num_classes, snapshot_path_to_restore=None, **params):
    print 'Setup Alexnet'
    net = tfext.alexnet.Alexnet(num_classes=num_classes, **params)
    with tf.variable_scope('lr'):
        conv_lr_pl = tf.placeholder(tf.float32, tuple(), name='conv_lr')
        fc_lr_pl = tf.placeholder(tf.float32, tuple(), name='fc_lr')

    loss_op = network_spec.loss(net.logits, net.y_gt)
    train_op = network_spec.training_convnet(net, loss_op,
                                             fc_lr=fc_lr_pl,
                                             conv_lr=conv_lr_pl)

    # Add the Op to compare the logits to the labels during correct_classified_top1.
    eval_correct_top1 = network_spec.correct_classified_top1(net.logits, net.y_gt)
    accuracy = tf.cast(eval_correct_top1, tf.float32) / \
               tf.constant(params['batch_size'], dtype=tf.float32)

    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph of the current sesion.
    tf.scalar_summary(['loss', 'batch_accuracy', 'conv_lr', 'fc_lr'], [loss_op, accuracy, conv_lr_pl, fc_lr_pl])

    net.sess.run(tf.initialize_all_variables())
    if snapshot_path_to_restore is not None:
        print('Restoring 7 layers from snapshot {}'.format(snapshot_path_to_restore))
        net.restore_from_snapshot(snapshot_path_to_restore, 7, restore_iter_counter=True)
    return net, train_op, loss_op, saver


def setup_convnet_network(num_classes, snapshot_path_to_restore=None, **params):
    print 'Setup Convnet'
    net = tfext.convnet.Convnet(num_classes=num_classes, **params)
    with tf.variable_scope('lr'):
        conv_lr_pl = tf.placeholder(tf.float32, tuple(), name='conv_lr')
        fc_lr_pl = tf.placeholder(tf.float32, tuple(), name='fc_lr')
    loss_op = network_spec.loss(net.logits, net.y_gt)
    train_op = network_spec.training_convnet(net, loss_op,
                                             fc_lr=fc_lr_pl,
                                             conv_lr=conv_lr_pl)

    # Add the Op to compare the logits to the labels during correct_classified_top1.
    eval_correct_top1 = network_spec.correct_classified_top1(net.logits, net.y_gt)
    accuracy = tf.cast(eval_correct_top1, tf.float32) / \
               tf.constant(params['batch_size'], dtype=tf.float32)

    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph of the current sesion.
    tf.scalar_summary(['loss', 'batch_accuracy', 'conv_lr', 'fc_lr'], [loss_op, accuracy, conv_lr_pl, fc_lr_pl])

    net.sess.run(tf.initialize_all_variables())
    # init with alexnet if necessary
    net.restore_from_alexnet_snapshot(trainhelper.get_alexnet_snapshot_path(),
                                      params['num_layers_to_init'])

    if snapshot_path_to_restore is not None:
        print('Restoring 5 layers from snapshot {}'.format(snapshot_path_to_restore))
        net.restore_from_snapshot(snapshot_path_to_restore, 5)
    return net, train_op, loss_op, saver


def eval_net(net, category, layer_names, mat_path, mean_path):
    roc_auc_dict = eval.olympicsports.roc. \
        roc_from_net.compute_roc_auc_from_net(net,
                                              category,
                                              layer_names,
                                              mat_path=mat_path,
                                              mean_path=mean_path,
                                              batch_size=256,
                                              norm_method=None)
    return roc_auc_dict


def eval_all_cat(net, step, global_step, summary_writer, params):
    aucs = {key: list() for key in params['test_layers']}
    for category in CATEGORIES:
        cur_cat_roc_dict = eval_net(net, category, params['test_layers'],
                                    mat_path=params['images_mat_pathes'][category],
                                    mean_path=params['mean_filepath'])

        print 'Step {}: {} ROCAUC = {}'.format(step, category, cur_cat_roc_dict)
        for layer_name, auc in cur_cat_roc_dict.iteritems():
            aucs[layer_name].append(auc)
            summary_writer.add_summary(
                tfext.utils.create_sumamry(
                    '{}ROCAUC_{}'.format(layer_name, category), auc),
                global_step=global_step)
    for layer_name, auc_list in aucs.iteritems():
        summary_writer.add_summary(
            tfext.utils.create_sumamry('m{}ROCAUC'.format(layer_name),
                                       np.mean(auc_list)),
            global_step=global_step)
    summary_writer.flush()


def run_training_current_clustering(net, batch_manager, train_op, loss_op,
                                    saver, **params):
    log_step = 1
    summary_step = 50

    summary_writer = tf.train.SummaryWriter(params['output_dir'], net.sess.graph)
    summary_op = tf.merge_all_summaries()

    for step in xrange(params['max_iter'] + 1):

        # test, snapshot
        if step == 1 or step % params['test_step'] == 0 or step + 1 == params['max_iter']:
            global_step = net.sess.run(net.global_iter_counter)
            eval_all_cat(net, step, global_step, summary_writer, params)
        if step % params['snapshot_iter'] == 0 and step > 1:
            checkpoint_prefix = os.path.join(params['output_dir'], 'checkpoint')
            saver.save(net.sess, checkpoint_prefix, global_step=global_step)
        if step == params['max_iter']:
            break

        # training
        start_time = time.time()
        feed_dict = batch_manager.fill_feed_dict(phase='train')
        feed_dict['lr/fc_lr:0'] = params['fc_lr']
        if step >= params['fix_conv_iter']:
            feed_dict['lr/conv_lr:0'] = params['conv_lr']
        else:
            feed_dict['lr/conv_lr:0'] = 0.0

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

        duration = time.time() - start_time

        if step % log_step == 0 or step + 1 == params['max_iter']:
            print('Step %d: loss = %.2f (%.3f s, %.2f im/s)'
                  % (step, loss_value, duration,
                     params['batch_size'] / duration))


def cluster_category(clustering_round, category, output_dir, params_clustering):
    # Delete old batch_ldr, recompute clustering and create new batch_ldr
    # Use HOGLDA for initial estimate of similarities
    # Just recluster on the HOGLDA sim every round
    print('Clustering %s' % category)
    # TODO: use RandomState with seed inside clustering
    np.random.seed(None)
    matrices = trainhelper.get_step_similarities(step=0,
                                                 net=None,
                                                 category=category,
                                                 dataset='OlympicSports',
                                                 layers=None,
                                                 pathtosim=params_clustering['pathtosim'],
                                                 pathtosim_avg=params_clustering[
                                                     'pathtosim_avg'])

    # Run clustering and update corresponding param fields
    params_clustering.update(matrices)
    params_clustering['clustering_round'] = clustering_round
    params_clustering['output_dir'] = output_dir
    # TODO: maybe recluster next round not on anchors
    index_dict, _ = trainhelper.run_reclustering(**params_clustering)
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
            if params['init_nbatches'] is None:
                params_clustering['init_nbatches'] = len(params_clustering['anchors']['anchor'])
            else:
                params_clustering['init_nbatches'] = params['init_nbatches']
            index_dict = cluster_category(clustering_round,
                                          cat,
                                          params['output_dir'],
                                          params_clustering)
            index_dict['labels'] = np.asarray(index_dict['labels']) + num_classes
            num_classes = index_dict['labels'].max() + 1

            batch_loader_params = params.copy()
            batch_loader_params['indexing_1_based'] = 0
            batch_loader_params['indexfile_path'] = index_dict
            batch_loader_params['images_mat_filepath'] = params['images_mat_pathes'][cat]
            batch_loaders[cat] = batch_loader_with_prefetch.BatchLoaderWithPrefetch(
                batch_loader_params)

        batch_manager = BatchManager(batch_loaders,
                                     params['batch_size'],
                                     params['im_shape'][:2],
                                     network=params['network'],
                                     random_shuffle=params['random_shuffle_categories'],
                                     random_seed=params['seed'])
        # Create network with new clustering parameters
        # If a network exists close session and reset graph
        if net is not None:
            net.sess.close()
        tf.reset_default_graph()
        if clustering_round == 0:
            snapshot_path_to_restore = params.pop('snapshot_path_to_restore')
        else:
            snapshot_path_to_restore = None
            assert 'snapshot_path_to_restore' not in params
        if params['network'] == tfext.alexnet.Alexnet:
            net, train_op, loss_op, saver = setup_alexnet_network(
                num_classes,
                snapshot_path_to_restore=snapshot_path_to_restore,
                **params)
        else:
            net, train_op, loss_op, saver = setup_convnet_network(
                num_classes,
                snapshot_path_to_restore=snapshot_path_to_restore,
                **params)

        # Restore from previous round model
        if clustering_round > 0:
            checkpoint_path = checkpoint_prefix + '-' + str(clustering_round)
            net.restore_from_snapshot(checkpoint_path, 7, restore_iter_counter=True)

        # Run training and save snapshot
        run_training_current_clustering(net, batch_manager, train_op, loss_op,
                                        saver, **params)
        saver.save(net.sess, checkpoint_prefix,
                   global_step=clustering_round + 1)
        batch_manager.cleanup()
        gc.collect()

    if net is not None:
        net.sess.close()
