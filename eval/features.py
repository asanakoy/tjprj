# Copyright (c) 2016 Artsiom Sanakoyeu

# pylint: disable=missing-docstring
from __future__ import division
import os.path
from os.path import join
import time

import tensorflow as tf
import numpy as np
import tfext.alexnet
import tfext.utils
import scipy.stats.mstats as stat
import scipy.spatial.distance as spdis
import sklearn
import scipy.io
from tqdm import tqdm


def compute_sim(net=None, norm_method='zscores', **params):
    accepable_methods = [None, 'zscores', 'unit_norm']
    if norm_method not in accepable_methods:
        raise ValueError('unknown norm method: {}. Use one of {}'.format(norm_method,
                                                                         accepable_methods))

    d = dict()
    d['features'] = extract_features(False, net=net, **params)
    d['features_flipped'] = extract_features(True, net=net, **params)
    stacked = dict()
    for key, val in d.iteritems():
        stacked[key] = np.hstack(val.values())
        if norm_method == 'zscores':
            stacked[key] = stat.zscore(stacked[key], axis=0)
        elif norm_method == 'unit_norm':
            # in-place
            sklearn.preprocessing.normalize(stacked[key], norm='l2', axis=1, copy=False)
        print 'Stacked {} shape: {}'.format(key, stacked[key].shape)

    sim_matrix = spdis.squareform(spdis.pdist(stacked['features'], metric='correlation'))
    sim_matrix_flip = spdis.cdist(stacked['features'], stacked['features_flipped'],
                                  metric='correlation')
    sim_matrix = np.float32(2 - sim_matrix)
    sim_matrix_flip = np.float32(2 - sim_matrix_flip)
    sim = {'simMatrix': sim_matrix, 'simMatrix_flip': sim_matrix_flip}
    return sim


def compute_sim_and_save(norm_method='zscores', net=None, **params):
    sim = compute_sim(norm_method=norm_method, net=net, **params)
    path_sim = join(params['sim_output_dir'], 'simMatrix_{}_{}_iter_{}_{}_{}.mat'.
                    format(params['category'], params['model_name'],
                           params['iter'], ''.join(params['layer_names']), norm_method))
    if not os.path.exists(params['sim_output_dir']):
        os.makedirs(params['sim_output_dir'])

    scipy.io.savemat(path_sim, sim)


def extract_features(flipped, net=None, **params):
    with tf.Graph().as_default():
        if net is None:
            is_temp_session = True
            net = tfext.alexnet.Alexnet(init_model=None, **params)
            saver = tf.train.Saver()
            net.sess.run(tf.initialize_all_variables())
            assert os.path.exists(params['snapshot_path'])
            saver.restore(net.sess, params['snapshot_path'])
        else:
            is_temp_session = False

        tensors_to_get = [net.__getattribute__(name) for name in params['layer_names']]
        print 'Tensors to extract:', tensors_to_get
        d = dict()
        total_num_images = params['image_getter'].total_num_images()
        for layer_name in params['layer_names']:
            tensor = net.__getattribute__(layer_name)
            d[layer_name] = np.zeros((total_num_images, np.prod(tensor.get_shape()[1:])))

        print 'Running {} iterations with batch_size={}'.\
            format(np.round(total_num_images / params['batch_size']), params['batch_size'])
        for step, batch_start in \
                tqdm(enumerate(range(0, total_num_images, params['batch_size']))):

            batch_idxs = range(batch_start,
                               min(batch_start + params['batch_size'], total_num_images))
            batch = params['image_getter'].get_batch(batch_idxs,
                                                     resize_shape=params['im_shape'],
                                                     mean=params['mean'])
            if flipped:
                batch = batch[:, :, ::-1, :]

            feed_dict = {
                net.x: batch,
                'input/is_phase_train:0': False
            }

            features = net.sess.run(tensors_to_get, feed_dict=feed_dict)
            for tensor_id, layer_name in enumerate(params['layer_names']):
                d[layer_name][batch_idxs, ...] = features[tensor_id].reshape(len(batch_idxs), -1)

        if is_temp_session:
            net.sess.close()
        return d
