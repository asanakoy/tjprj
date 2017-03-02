import tensorflow as tf

import os
from os.path import join
import numpy as np
from eval.image_getter import ImageGetterFromMat
import eval.olympicsports.roc.roc_auc
import eval.features
import eval.olympicsports.utils
from utils import get_sim_pathes


if __name__ == '__main__':
    category = ''


    mat_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/' + category + '/images.mat'
    model_name = 'alexnet_joint_categories_begin'

    category = ''
    mean_path = join(
        '/export/home/asanakoy/workspace/OlympicSports/cnn/', model_name, category, 'mean.npy')
    # mean = np.load(mean_path)

    params = {
        'category': category,
        'layer_names': ['fc7'],
        'norm_method': None,
        'image_getter': ImageGetterFromMat(mat_path),
        'mean': None,
        'im_shape': (227, 227),
        'batch_size': 256,
        'use_batch_norm': False,
        'gpu_memory_fraction': 0.35,
        'device_id': '/gpu:{}'.format(0)
    }
    snapshot_path, sim_output_path = get_sim_pathes(model_name,iteration=170004,round_id=None,
                                                              **params)
    net = eval.features.load_net_with_graph(snapshot_path, gpu_memory_fraction=0.4,
                               conv5='conv5/conv5',
                               maxpool5='conv5/maxpool:0',
                               fc6='fc6/fc:0',
                               fc7='fc7/fc:0'
                               )

    print 'Using Snapshot:', snapshot_path
    print 'Output sim matrix to', sim_output_path
    eval.features.compute_sim_and_save(sim_output_path, net=net, **params)
    net.sess.close()
    print params['layer_names']
    # WARNING! Don't forget to change to the category variable when testing non-joint models!
    eval.olympicsports.roc.roc_auc.compute_roc_auc_from_sim(['long_jump'], path_sim_matrix=sim_output_path)
