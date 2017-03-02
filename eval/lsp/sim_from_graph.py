import tensorflow as tf
from collections import namedtuple
import os
from os.path import join
import numpy as np
from eval.image_getter import ImageGetterFromPaths
import eval.olympicsports.roc.roc_auc
import eval.features
import eval.olympicsports.utils
import glob

Net = namedtuple('Net', ['sess', 'fc7', 'fc6', 'maxpool5', 'conv5', 'graph'])


def get_sim_output_path(model_name, iteration, params):
    p = join('/export/home/asanakoy/workspace/lsp/sim/tf',
             params['category'], 'simMatrix_{}_{}{}_iter_{}_{}_{}.mat'.
             format(params['category'], model_name,
                    '_nomean' if params['mean'] is None else '',
                    iteration, ''.join(params['layer_names']), params['norm_method']))
    return p


if __name__ == '__main__':

    for iteration in [180000]:
        crops_dir = '/export/home/asanakoy/workspace/lsp/crops_227x227'
        image_paths = [join(crops_dir, p) for p in glob.glob1(crops_dir, '*.png')]
        im_shape = (227, 227)
        model_name = 'convnet_joint_categories_scratch'
        # iteration = 80002
        snapshot_path = join('/export/home/asanakoy/workspace/OlympicSports/cnn/',
                               model_name, '', 'checkpoint-{}'.format(iteration))

        category = ''
        mean_path = join(
            '/export/home/asanakoy/workspace/OlympicSports/cnn/', model_name, category, 'mean.npy')
        mean = np.load(mean_path)
        for layer_name in ['maxpool5', 'fc6']:
            params = {
                'category': '',
                'layer_names': [layer_name],
                'norm_method': None,
                'image_getter': ImageGetterFromPaths(image_paths, im_shape),
                'mean': None,
                'im_shape': (227, 227),
                'batch_size': 256,
                'use_batch_norm': False,
                'gpu_memory_fraction': 0.39,
                'device_id': '/gpu:{}'.format(0)
            }

            sim_output_path = get_sim_output_path(model_name, iteration, params)

            net = eval.features.load_net_with_graph(snapshot_path, gpu_memory_fraction=0.4,
                                      conv5='conv5/conv5:0',
                                      maxpool5='maxpool5:0',
                                      fc6='fc6/fc6:0',
                                      )

            print 'Using Snapshot:', snapshot_path
            print 'Output sim matrix to', sim_output_path
            eval.features.compute_sim_and_save(sim_output_path, net=net, **params)
            net.sess.close()
            print params['layer_names']
