# Copyright (c) 2016 Artsiom Sanakoyeu

# pylint: disable=missing-docstring
import os.path
from os.path import join
import time
import sys
import numpy as np
from tfeval.image_getter import ImageGetterFromMat
import tfeval.features
import tfeval.olympicsports.utils


def get_pathes(model_name, iteration=0, round_id=0, **params):
    if round_id is None:
        init_model_path = join('/export/home/asanakoy/workspace/OlympicSports/cnn/',
                               model_name,
                               params['category'], 'checkpoint-{}'.format(iteration))

        sim_output_path = join('/export/home/mbautist/Desktop/',
                               params['category'], 'simMatrix_{}_{}_iter_{}_{}_{}.mat'.
                               format(params['category'], model_name, iteration,
                                      ''.join(params['layer_names']), params['norm_method']))
    else:
        init_model_path = join('/export/home/mbautist/tmp/tf_test/Caltech101/checkpoint-1')

        sim_output_path = join('/export/home/mbautist/Desktop/',
                               params['category'], 'simMatrix_{}_{}_rounds_{}_{}_{}.mat'.
                               format(params['category'], round_id, model_name,
                                      ''.join(params['layer_names']), params['norm_method']))
    return init_model_path, sim_output_path


def main(argv):
    if len(argv) > 1:
        category = argv[1]
    else:
        category = 'Caltech101'
    model_name = 'iter1'
    is_bbox_sq = 0

    mat_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/Caltech101/crops/Caltech101/images_mat.mat'


    if is_bbox_sq:
        mean_path = join('/export/home/asanakoy/workspace/OlympicSports/cnn', model_name, category, 'mean.npy')
    else:
        mean_path = join(
            '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' + category + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/mean.npy')
    mean = np.load(mean_path)

    params = {
        'category': category,
        'number_layers_restore': 7,
        'layer_names': ['fc7'],
        'norm_method': 'zscores',
        'image_getter': ImageGetterFromMat(mat_path),
        'mean': mean,
        'im_shape': (227, 227),
        'batch_size': 256,
        'use_batch_norm': False,
        'gpu_memory_fraction': 0.35,
        'device_id': '/gpu:{}'.format(0)
    }
    params['snapshot_path'], sim_output_path = get_pathes(model_name,
                                                          iteration=40000,
                                                          round_id=2,
                                                          **params)
    print 'Using Snapshot:', params['snapshot_path']
    print 'Output sim matrix to', sim_output_path
    tfeval.features.compute_sim_and_save(sim_output_path, **params)


if __name__ == '__main__':
    main(sys.argv[1:])
