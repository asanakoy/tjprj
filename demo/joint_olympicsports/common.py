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
import tfext.fcconvnetv2
import tfext.utils
from trainhelper import trainhelper
import batch_loader_with_prefetch
import tfeval.olympicsports.roc.roc_from_net
import tfeval.olympicsports.utils
import helper
from helper import BatchManager
from tqdm import tqdm
import pickle
import copy

from clustering.batchgenerator import BatchGenerator
from clustering.batchsampler import BatchSampler


def get_first_model_path():
    return '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.npy'


def training_one_layer(net, loss_op, layer_name, lr, optimizer_type='adagrad'):
    with net.graph.as_default():
        print('Creating optimizer {}'.format(optimizer_type))
        if optimizer_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(lr, initial_accumulator_value=0.0001)
        elif optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif optimizer_type == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)
        else:
            raise ValueError('Unknown optimizer type {}'.format(optimizer_type))

        layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name)
        grads = tf.gradients(loss_op, layer_vars)

        train_op = optimizer.apply_gradients(zip(grads, layer_vars),
                                                  global_step=net.global_iter_counter,
                                                  name='last_layer_train_op')
        return train_op


def run_reclustering(clustering_round=None,

                     num_initial_batches=None,
                     sim_matrix=None,
                     flipvals=None,
                     seq_names=None,
                     crops_dir=None,
                     relative_image_pathes=None,
                     num_cliques_per_initial_batch=None,
                     num_samples_per_clique=None,
                     anchors=None,

                     batch_size=None,
                     num_batches_to_sample=None,
                     max_cliques_per_batch=None,
                     output_dir=None,
                     seed=None):
    """
    Run clustering assignment procedure and return arrays for BatchLoader in a dict
    :param kwargs_generator: arguments for generator
    :param kwargs_sampler: arguments for sampler
    :return: Dict of arrays for BatchLoader
    """
    generator = BatchGenerator(sim_matrix=sim_matrix,
                               flipvals=flipvals,
                               seq_names=seq_names,
                               relative_image_pathes=relative_image_pathes,
                               crops_dir=crops_dir,
                               num_cliques_per_initial_batch=num_cliques_per_initial_batch,
                               num_samples_per_clique=num_samples_per_clique,
                               anchors=anchors,
                               seed=seed)
    init_batches = generator.generate_batches(num_initial_batches=num_initial_batches)
    sampler = BatchSampler(batches=init_batches,
                           sim_matrix=sim_matrix,
                           flipvals=flipvals,
                           seq_names=seq_names,
                           crops_dir=crops_dir,
                           relative_image_pathes=relative_image_pathes,
                           seed=seed)

    # # Save batchsampler
    sampler_file = open(os.path.join(output_dir, 'sampler_round_' + str(clustering_round) + '.pkl'), 'wb')
    pickle.dump(sampler.cliques, sampler_file, pickle.HIGHEST_PROTOCOL)
    sampler_file.close()

    indices = np.empty(0, dtype=np.int64)
    flipped = np.empty(0, dtype=np.bool)
    label = np.empty(0, dtype=np.int64)
    print 'Sampling batches'
    for i in tqdm(range(num_batches_to_sample)):
        # print "Sampling batch {}".format(i)
        batch = sampler.sample_batch(batch_size,
                                     max_cliques_per_batch,
                                     mode='random')
        _x, _f, _y = sampler.parse_to_list(batch)
        assert len(_x) == len(_f) == len(_y) == batch_size
        indices = np.append(indices, _x.astype(dtype=np.int64))
        flipped = np.append(flipped, _f.astype(dtype=np.bool))
        label = np.append(label, _y.astype(dtype=np.int64))

    assert indices.shape[0] == flipped.shape[0] == label.shape[
        0], "Corrupted arguments for batch loader"
    return {'idxs': indices, 'flipvals': flipped, 'labels': label}


def set_summary_tracking(graph, track_moving_averages):
    with graph.as_default():
        if track_moving_averages:
            movin_avg_vars = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES) if 'moving_' in v.name]
            for v in movin_avg_vars:
                tf.summary.scalar('moving/' + v.name, tf.nn.l2_loss(v))


def create_loss_op(net, loss_type):
    if loss_type not in ['clique', 'soft_xe']:
        raise ValueError('Unknown loss_type: {}'.format(loss_type))
    if loss_type == 'soft_xe':
        print 'Creating soft XE loss'
        features_normed = tf.nn.l2_normalize(net.fc7, dim=1, name='normalized_features')
        _, labels = tf.unique(net.y_gt, name='unique_labels')
        loss_op = tfext.network_spec.soft_xe(features_normed, labels, alpha=1.0, num_classes_in_batch=8, sess=None)
    else:
        loss_op = tfext.network_spec.loss(net.logits, net.y_gt)
    return loss_op


def setup_alexnet_network(num_classes, loss_type, batch_size, optimizer_type,
                          snapshot_path_to_restore=None, net_params=None):
    print 'Setup Alexnet'
    net = tfext.alexnet.Alexnet(num_classes=num_classes, **net_params)
    with tf.variable_scope('lr'):
        conv_lr_pl = tf.placeholder(tf.float32, tuple(), name='conv_lr')
        fc_lr_pl = tf.placeholder(tf.float32, tuple(), name='fc_lr')

    loss_op = create_loss_op(net, loss_type=loss_type)
    train_op = network_spec.training_convnet(net, loss_op,
                                             fc_lr=fc_lr_pl,
                                             conv_lr=conv_lr_pl,
                                             optimizer_type=optimizer_type)
    if loss_type != 'soft_xe':
        training_one_layer(net, loss_op, 'fc8', fc_lr_pl)

    # Add the Op to compare the logits to the labels during correct_classified_top1.
    eval_correct_top1 = network_spec.correct_classified_top1(net.logits, net.y_gt)
    accuracy = tf.cast(eval_correct_top1, tf.float32) / \
               tf.constant(batch_size, dtype=tf.float32)

    saver = tf.train.Saver(max_to_keep=100)

    conv5w_norm = tf.nn.l2_loss(net.graph.get_tensor_by_name('conv5/weight:0'))
    conv5b_norm = tf.nn.l2_loss(net.graph.get_tensor_by_name('conv5/bias:0'))
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('batch_accuracy', accuracy)
    tf.summary.scalar('conv_lr', conv_lr_pl)
    tf.summary.scalar('fc_lr', fc_lr_pl)
    tf.summary.scalar('conv5w_norm', conv5w_norm)
    tf.summary.scalar('conv5b_norm', conv5b_norm)

    net.sess.run(tf.global_variables_initializer())
    if snapshot_path_to_restore is not None:
        print('Restoring 7 layers from snapshot {}'.format(snapshot_path_to_restore))
        net.restore_from_snapshot(snapshot_path_to_restore, 7, restore_iter_counter=True)
    return net, train_op, loss_op, saver


def setup_convnet_network(network_class, num_classes, loss_type, batch_size,
                          optimizer_type, snapshot_path_to_restore=None, net_params=None):
    print 'Setup Convnet'
    if network_class not in [tfext.convnet.Convnet, tfext.fcconvnetv2.FcConvnetV2]:
        raise ValueError('Unknown network')

    net = network_class(num_classes=num_classes, **net_params)
    with tf.variable_scope('lr'):
        conv_lr_pl = tf.placeholder(tf.float32, tuple(), name='conv_lr')
        fc_lr_pl = tf.placeholder(tf.float32, tuple(), name='fc_lr')
    loss_op = create_loss_op(net, loss_type=loss_type)

    # Add the Op to compare the logits to the labels during correct_classified_top1.
    eval_correct_top1 = network_spec.correct_classified_top1(net.logits, net.y_gt)
    accuracy = tf.cast(eval_correct_top1, tf.float32) / \
               tf.constant(batch_size, dtype=tf.float32)

    saver = tf.train.Saver(max_to_keep=100)

    conv5w_norm = tf.nn.l2_loss(net.graph.get_tensor_by_name('conv5/weight:0'))
    conv5b_norm = tf.nn.l2_loss(net.graph.get_tensor_by_name('conv5/bias:0'))
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('batch_accuracy', accuracy)
    tf.summary.scalar('conv_lr', conv_lr_pl)
    tf.summary.scalar('fc_lr', fc_lr_pl)
    tf.summary.scalar('conv5w_norm', conv5w_norm)
    tf.summary.scalar('conv5b_norm', conv5b_norm)

    net.sess.run(tf.global_variables_initializer())
    # init with alexnet if necessary
    net.restore_from_alexnet_snapshot(trainhelper.get_alexnet_snapshot_path(),
                                      net_params['num_layers_to_init'])
    if snapshot_path_to_restore is not None:
        if network_class == tfext.convnet.Convnet:
            num_layers_to_restore = 5
        else:
            num_layers_to_restore = 7
        print('Restoring {} layers from snapshot {}'.format(num_layers_to_restore, snapshot_path_to_restore))
        net.restore_from_snapshot(snapshot_path_to_restore, num_layers_to_restore, restore_iter_counter=True)

    train_op = network_spec.training_convnet(net, loss_op,
                                             fc_lr=fc_lr_pl,
                                             conv_lr=conv_lr_pl,
                                             optimizer_type=optimizer_type,
                                             trace_gradients=True)
    last_layer_name = 'fc6' if network_class == tfext.convnet.Convnet else 'fc8'
    training_one_layer(net, loss_op, last_layer_name, fc_lr_pl)

    uninit_vars = [v for v in tf.global_variables()
                   if not tf.is_variable_initialized(v).eval(session=net.sess)]
    print 'uninit vars:', [v.name for v in uninit_vars]
    assert optimizer_type == 'sgd' or len(uninit_vars) > 0
    net.sess.run(tf.variables_initializer(uninit_vars))

    return net, train_op, loss_op, saver


def eval_net(net, category, layer_names, mat_path, mean_path):
    roc_auc_dict = tfeval.olympicsports.roc. \
        roc_from_net.compute_roc_auc_from_net(net,
                                              category,
                                              layer_names,
                                              mat_path=mat_path,
                                              mean_path=mean_path,
                                              batch_size=256,
                                              norm_method=None)
    return roc_auc_dict


def eval_all_cat(net, step, global_step, summary_writer, params):
    CATEGORIES = params['categories']
    aucs = {key: list() for key in params['test_layers']}
    for category in CATEGORIES:
        cur_cat_roc_dict = eval_net(net, category, params['test_layers'],
                                    mat_path=params['images_mat_pathes'][category],
                                    mean_path=params['mean_filepath'])

        print 'Step {}: {} ROCAUC = {}'.format(global_step, category, cur_cat_roc_dict)
        for layer_name, auc in cur_cat_roc_dict.iteritems():
            aucs[layer_name].append(auc)
            summary_writer.add_summary(
                tfext.utils.create_sumamry(
                    '{}ROCAUC_{}'.format(layer_name, category), auc),
                global_step=global_step)
    scores = dict()
    for layer_name, auc_list in aucs.iteritems():
        cur_layer_mean = np.mean(auc_list)
        scores[layer_name] = cur_layer_mean
        summary_writer.add_summary(
            tfext.utils.create_sumamry('m{}ROCAUC'.format(layer_name),
                                       cur_layer_mean),
            global_step=global_step)
    summary_writer.flush()
    return scores


def run_training_current_clustering(net, batch_manager, train_op, loss_op,
                                    saver, clustering_round=None, **params):
    log_step = 1
    summary_step = 50

    with net.graph.as_default():
        summary_writer = tf.summary.FileWriter(params['output_dir'], net.sess.graph)
        summary_op = tf.summary.merge_all()
        fc_train_op = net.graph.get_operation_by_name('fc_train_op')
        try:
            last_layer_train_op = net.graph.get_operation_by_name('last_layer_train_op')
        except KeyError:
            last_layer_train_op = None

    best_scores = {key: 0.0 for key in params['test_layers']}

    for step in xrange(params['max_iter'] + 1):

        # test, snapshot
        if step == 1 or step % params['test_step'] == 0 or step + 1 == params['max_iter']:
            global_step = net.sess.run(net.global_iter_counter)
            scores = eval_all_cat(net, step, global_step, summary_writer, params)
            if params['snapshot_iter'] == 'save_the_best':
                should_save_snapshot = False
                for layer_name, layer_auc in scores.iteritems():
                    if layer_auc - best_scores[layer_name] >= 0.001:
                        print 'FOUND THE BEST SCORE of layer {}!'.format(layer_name)
                        best_scores[layer_name] = layer_auc
                        should_save_snapshot = True
                        open(os.path.join(params['output_dir'],
                             '{}_{:.3f}_iter-{}'.format(layer_name, layer_auc, global_step)), mode='w').close()
                if should_save_snapshot:
                    checkpoint_prefix = os.path.join(params['output_dir'], 'checkpoint')
                    saver.save(net.sess, checkpoint_prefix, global_step=global_step)

        if params['snapshot_iter'] != 'save_the_best' and \
                step % params['snapshot_iter'] == 0 and step > 0:
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

        if step < params['fix_up_to_the_last']:
            cur_train_op = last_layer_train_op
        elif step < params.get('only_fc_train_op_iter', 0):
            cur_train_op = fc_train_op
        else:
            cur_train_op = train_op

        if step % summary_step == 0 or step + 1 == params['max_iter'] or step % params['test_step'] == 0:
            global_step, summary_str, _, loss_value = net.sess.run(
                [net.global_iter_counter,
                 summary_op,
                 cur_train_op,
                 loss_op],
                feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step)
        else:
            global_step, _, loss_value = net.sess.run(
                [net.global_iter_counter, cur_train_op, loss_op],
                feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % log_step == 0 or step + 1 == params['max_iter']:
            print('Step %d: loss = %.2f (%.3f s, %.2f im/s)'
                  % (global_step, loss_value, duration,
                     params['batch_size'] / duration))


def cluster_category(clustering_round=None,
                     recluster_on_init_sim=True,

                     num_initial_batches=None,
                     pathtosim=None,
                     pathtosim_avg=None,  # only used for clustering on HOG-LDA
                     seq_names=None,
                     crops_dir=None,
                     relative_image_pathes=None,
                     num_cliques_per_initial_batch=None,
                     num_samples_per_clique=None,
                     anchors=None,

                     batch_size=None,
                     num_batches_to_sample=None,
                     max_cliques_per_batch=None,
                     output_dir=None,
                     seed=None,

                     dataset='OlympicSports',
                     category=''):
    # Delete old batch_ldr, recompute clustering and create new batch_ldr
    # Use HOGLDA for initial estimate of similarities
    # Just recluster on the HOGLDA sim every round
    print('Clustering %s' % category)
    matrices = trainhelper.get_step_similarities(step=0 if recluster_on_init_sim else clustering_round,
                                                 net=None,
                                                 category=category,
                                                 dataset=dataset,
                                                 layers=None,
                                                 pathtosim=pathtosim,
                                                 pathtosim_avg=pathtosim_avg)

    # Run clustering and update corresponding param fields
    # TODO: maybe recluster next round not on anchors
    if seed is not None:
        seed += clustering_round

    index_dict = run_reclustering(clustering_round=None,

                                  num_initial_batches=num_initial_batches,
                                  sim_matrix=matrices['sim_matrix'],
                                  flipvals=matrices['flipvals'],
                                  seq_names=seq_names,
                                  crops_dir=crops_dir,
                                  relative_image_pathes=relative_image_pathes,
                                  num_cliques_per_initial_batch=num_cliques_per_initial_batch,
                                  num_samples_per_clique=num_samples_per_clique,
                                  anchors=anchors,

                                  batch_size=batch_size,
                                  num_batches_to_sample=num_batches_to_sample,
                                  max_cliques_per_batch=max_cliques_per_batch,
                                  output_dir=output_dir,
                                  seed=seed)
    return index_dict


def run_training(**params):
    net = None
    CATEGORIES = params.setdefault('categories', helper.ALL_CATEGORIES)
    params.setdefault('loss_type', 'clique')
    params.setdefault('optimizer_type', 'adagrad')
    params.setdefault('reset_iter_counter', False)
    params.setdefault('recluster_on_init_sim', True)
    checkpoint_prefix = os.path.join(params['output_dir'], 'round_checkpoint')
    tfeval.olympicsports.utils.get_joint_categories_mean(params['mean_filepath'], CATEGORIES)
    assert os.path.exists(params['mean_filepath'])

    for clustering_round in range(0, params['num_clustering_rounds']):

        num_classes = 0
        batch_loaders = dict()
        for cat in CATEGORIES:
            params_clustering = trainhelper.get_default_params_clustering('OlympicSports', cat)
            del params_clustering['category']
            for key in params['custom_params_clustering']:
                if key not in params_clustering:
                    raise ValueError('Unexpected key in custom_params_clustering: {}'.format(key))
            params_clustering.update(params['custom_params_clustering'])
            # set num batches of cliques to number of anchors
            if params_clustering['num_initial_batches'] is None:
                params_clustering['num_initial_batches'] = len(params_clustering['anchors']['anchor'])
            index_dict = cluster_category(clustering_round,
                                          recluster_on_init_sim=params['recluster_on_init_sim'],
                                          category=cat,
                                          output_dir=params['output_dir'],
                                          seed=params['seed'], **params_clustering)
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
                                     categories=CATEGORIES,
                                     random_shuffle=params['random_shuffle_categories'],
                                     random_seed=params['seed'])
        # Create network with new clustering parameters
        # If a network exists close session and reset graph
        if net is not None:
            net.sess.close()
            # with net.graph.as_default():
            #     tf.reset_default_graph()
        if clustering_round == 0:
            snapshot_path_to_restore = params.pop('snapshot_path_to_restore')
        else:
            snapshot_path_to_restore = None
            assert 'snapshot_path_to_restore' not in params

        with tf.Graph().as_default():
            if params['network'] == tfext.alexnet.Alexnet:
                net, train_op, loss_op, saver = setup_alexnet_network(
                    num_classes,
                    params['loss_type'],
                    params['batch_size'],
                    optimizer_type=params['optimizer_type'],
                    snapshot_path_to_restore=snapshot_path_to_restore,
                    net_params=params)
            elif params['network'] == tfext.convnet.Convnet or params['network'] == tfext.fcconvnetv2.FcConvnetV2:
                net, train_op, loss_op, saver = setup_convnet_network(
                    params['network'],
                    num_classes,
                    params['loss_type'],
                    params['batch_size'],
                    optimizer_type=params['optimizer_type'],
                    snapshot_path_to_restore=snapshot_path_to_restore,
                    net_params=params)
            else:
                raise ValueError('Unknown network')
            if clustering_round == 0 and params['reset_iter_counter']:
                tf.variables_initializer([net.global_iter_counter])

        set_summary_tracking(net.graph, track_moving_averages=params.get('track_moving_averages', False))

        # Restore from previous round model
        if clustering_round > 0:
            checkpoint_path = checkpoint_prefix + '-' + str(clustering_round)
            num_layers_to_restore = 7 if \
                params['network'] == tfext.alexnet.Alexnet or \
                params['network'] == tfext.fcconvnetv2.FcConvnetV2 else 5
            net.restore_from_snapshot(checkpoint_path, num_layers_to_restore, restore_iter_counter=True)

        with net.graph.as_default():
            print 'Reset momentum and Adagrad accumulators'
            optimizer_vars = [v for v in tf.global_variables()
                           if 'adagrad' in v.name.lower() or 'momentum' in v.name.lower()]
            print [v.name for v in optimizer_vars]
            tf.variables_initializer(optimizer_vars)

        # Run training and save snapshot
        run_training_current_clustering(net, batch_manager, train_op, loss_op,
                                        saver, clustering_round=clustering_round, **params)
        saver.save(net.sess, checkpoint_prefix,
                   global_step=clustering_round + 1)
        batch_manager.cleanup()
        gc.collect()

    if net is not None:
        net.sess.close()
