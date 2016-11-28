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
import sklearn.preprocessing
import scipy.io
import math
from tqdm import tqdm


def compute_sim(net=None, norm_method='zscores', return_features=False, **params):
    """
    Args:
        net: If specified use this network for feature extraction
        norm_method: features normalization method. Do not normalize if None.
                     Must be one of [None, 'zscores', unit_norm]
        return_features: return tuple (sim, features) if true. Otherwise return sim.
        params: other parameters
    """
    accepable_methods = [None, 'zscores', 'unit_norm']
    if norm_method not in accepable_methods:
        raise ValueError('unknown norm method: {}. Use one of {}'.format(norm_method,
                                                                         accepable_methods))

    features_dict = dict()
    features_dict['features'] = extract_features(False, net=net, **params)
    features_dict['features_flipped'] = extract_features(True, net=net, **params)
    stacked = dict()
    for key, val in features_dict.iteritems():
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
    sim = {'sim_matrix': sim_matrix, 'simMatrix_flip': sim_matrix_flip}

    if return_features:
        return sim, features_dict
    else:
        return sim


def compute_sim_and_save(sim_output_path, norm_method, net=None, **params):
    sim = compute_sim(norm_method=norm_method, net=net, **params)
    if not os.path.exists(os.path.dirname(sim_output_path)):
        os.makedirs(os.path.dirname(sim_output_path))
    scipy.io.savemat(sim_output_path, sim)


def extract_features(flipped, net=None, frame_ids=None, **params):
    """
    Extract features from the network.

    Args:
        flipped: are frames flipped?
        net: if not None it will use this network to extract the features,
          otherwise create new net and restore from teh snapshot
        frame_ids: if None extract from all frames,
          if a list of frames - extract features only for them
        params:
    """
    if not isinstance(params['layer_names'], list):
        raise TypeError('layer_names must be a list')

    if net is None:
        if 'number_layers_restore' not in params:
            raise KeyError('number_layers_restore must be in params if net is None')
        is_temp_session = True
        with tf.Graph().as_default():
            print 'Creating default Alexnet Network with {} learned layers'.format(params['number_layers_restore'])
            if params['number_layers_restore'] == 8 and 'num_classes' not in params:
                raise ValueError('You must specify "num_classes" if you restore 8 layers')
            net = tfext.alexnet.Alexnet(init_model=None, **params)
            net.sess.run(tf.initialize_all_variables())
            assert os.path.exists(params['snapshot_path'])
            net.restore_from_snapshot(params['snapshot_path'], params['number_layers_restore'])
    else:
        is_temp_session = False

    tensors_to_get = [net.__getattribute__(name) for name in params['layer_names']]
    print 'Tensors to extract:', tensors_to_get
    d = dict()
    if frame_ids is None:
        frame_ids = np.arange(params['image_getter'].total_num_images())

    for layer_name in params['layer_names']:
        tensor = net.__getattribute__(layer_name)
        d[layer_name] = np.zeros((len(frame_ids), np.prod(tensor.get_shape()[1:])))

    num_batches = int(math.ceil(len(frame_ids) / params['batch_size']))
    print 'Running {} iterations with batch_size={}'.\
        format(num_batches, params['batch_size'])
    for step, batch_start in tqdm(enumerate(range(0, len(frame_ids), params['batch_size'])),
                                  total=num_batches):
        batch_idxs = frame_ids[batch_start:batch_start + params['batch_size']]
        batch = params['image_getter'].get_batch(batch_idxs,
                                                 resize_shape=params['im_shape'],
                                                 mean=params['mean'])
        if flipped:
            batch = batch[:, :, ::-1, :]

        feed_dict = {
            'input/x:0': batch,
            'input/is_phase_train:0': False
        }

        features = net.sess.run(tensors_to_get, feed_dict=feed_dict)
        pos_begin = params['batch_size'] * step
        pos_end = pos_begin + len(batch_idxs)
        for tensor_id, layer_name in enumerate(params['layer_names']):
            d[layer_name][pos_begin:pos_end, ...] = features[tensor_id].reshape(len(batch_idxs), -1)

    if is_temp_session:
        net.sess.close()
    return d
