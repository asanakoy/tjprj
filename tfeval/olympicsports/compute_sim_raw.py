# Copyright (c) 2016 Artsiom Sanakoyeu

# pylint: disable=missing-docstring
import tfeval.features
import tfeval.olympicsports.utils
import tensorflow as tf
from collections import namedtuple
import os
import numpy as np
from tfeval.image_getter import ImageGetterFromMat
import tfeval.olympicsports.roc.roc_auc
import tfeval.features
import tfeval.olympicsports.utils
from utils import get_sim_pathes

Net = namedtuple('Net', ['sess', 'fc7', 'fc6', 'maxpool5', 'conv5', 'graph'])


def load_net(snapshot_path, gpu_memory_fraction=None,
             conv5=None,
             maxpool5=None,
             fc6=None,
             fc7=None):
    graph_path = snapshot_path + '.meta'
    if not os.path.exists(graph_path):
        raise IOError('Graph meta file not found: {}'.format(graph_path))
    if not os.path.exists(snapshot_path):
        raise IOError('Snapshot file not found: {}'.format(snapshot_path))

    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    if gpu_memory_fraction is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=config)
        new_saver = tf.train.import_meta_graph(graph_path)
        new_saver.restore(sess, snapshot_path)

        return Net(sess=sess, graph=graph,
                   conv5=None if conv5 is None else tf.get_default_graph().get_tensor_by_name(
                       conv5),
                   maxpool5=None if maxpool5 is None else tf.get_default_graph().get_tensor_by_name(
                       maxpool5),
                   fc6=None if fc6 is None else tf.get_default_graph().get_tensor_by_name(fc6),
                   fc7=None if fc7 is None else tf.get_default_graph().get_tensor_by_name(fc7))


def main(gpu_id, category):

    mat_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/' + category + '/images.mat'
    mean_path = '/export/home/asanakoy/workspace/OlympicSports/cnn/convnet_noaug_convlr0.01_fclr0.01_warmup2000/' + category + '/mean.npy'
    mean = np.load(mean_path)

    # model_names = ['alexnet']
    model_names = ['convnet_noaug_convlr0.01_fclr0.01_warmup2000']
    for model_name in model_names:
        for lr_name in ['maxpool5', 'fc6']:
            for mean_value in [mean]:

                if lr_name == 'fc6':
                    norm_method = 'zscores'
                else:
                    norm_method = None

                params = {
                    'category': category,
                    'layer_names': [lr_name],
                    'norm_method': norm_method,
                    'image_getter': ImageGetterFromMat(mat_path),
                    'mean': mean_value,
                    'im_shape': (227, 227),
                    'batch_size': 256,
                    'use_batch_norm': False,
                    'gpu_memory_fraction': 0.35,
                    'device_id': '/gpu:{}'.format(gpu_id)
                }
                snapshot_path, sim_output_path = get_sim_pathes(model_name,
                                                                iteration=None,
                                                                round_id=1,
                                                                **params)
                net = load_net(snapshot_path, gpu_memory_fraction=0.4,
                               conv5='conv5/conv5:0',
                               maxpool5='maxpool5:0',
                               fc6='fc6/fc6:0',
                               # fc7='fc7/fc:0'
                               )
                print 'Using Snapshot:', snapshot_path
                print 'Output sim matrix to', sim_output_path
                tfeval.features.compute_sim_and_save(sim_output_path, net=net, **params)


if __name__ == '__main__':
    # main(0, 'long_jump')

    CATEGORIES = [
                  'basketball_layup',
                  'bowling',
                  'clean_and_jerk',
                  'discus_throw',
                  'diving_platform_10m',
                  'diving_springboard_3m',
                  'hammer_throw',
                  'high_jump',
                  'javelin_throw',
                  'long_jump',
                  'pole_vault',
                  'shot_put',
                  'snatch',
                  'tennis_serve',
                  'triple_jump',
                  'vault']

    for cat in CATEGORIES:
        main(0, cat)
