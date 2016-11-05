# Copyright (c) 2016 Artsiom Sanakoyeu

# pylint: disable=missing-docstring
import os.path
from os.path import join
import time
import sys
import numpy as np
from eval.image_getter import ImageGetterFromMat
import eval.features
import eval.olympicsports.utils
from utils import get_sim_pathes


def main(gpu_id, category):
    mat_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/' + category + '/images.mat'
    model_name = 'convnet_scratch'

    mean_path = join('/export/home/asanakoy/workspace/OlympicSports/cnn/', model_name, category, 'mean.npy')
    mean = np.load(mean_path)

    params = {
        'category': category,
        'number_layers_restore': 6,
        'layer_names': ['fc6'],
        'norm_method': None,
        'image_getter': ImageGetterFromMat(mat_path),
        'mean': mean,
        'im_shape': (227, 227),
        'batch_size': 256,
        'use_batch_norm': False,
        'gpu_memory_fraction': 0.35,
        'device_id': '/gpu:{}'.format(gpu_id)
    }
    params['snapshot_path'], sim_output_path = get_sim_pathes(model_name,
                                                              iteration=None,
                                                              round_id=1,
                                                              **params)
    print 'Using Snapshot:', params['snapshot_path']
    print 'Output sim matrix to', sim_output_path
    eval.features.compute_sim_and_save(sim_output_path, **params)


if __name__ == '__main__':
    main(0, 'long_jump')

    # CATEGORIES = [
    #               'basketball_layup',
    #               'bowling',
    #               'clean_and_jerk',
    #               'discus_throw',
    #               'diving_platform_10m',
    #               'diving_springboard_3m',
    #               'hammer_throw',
    #               'high_jump',
    #               'javelin_throw',
    #               'long_jump',
    #               'pole_vault',
    #               'shot_put',
    #               'snatch',
    #               'tennis_serve',
    #               'triple_jump',
    #               'vault']
    #
    # for cat in CATEGORIES:
    #     main(0, cat)
