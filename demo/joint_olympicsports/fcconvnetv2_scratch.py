# Original work Copyright 2015 The TensorFlow Authors. Licensed under the Apache License v2.0 http://www.apache.org/licenses/LICENSE-2.0
# Modified work Copyright (c) 2016 Artsiom Sanakoyeu

# pylint: disable=missing-docstring
# from __future__ import absolute_import
from __future__ import division

import os.path
import time
import datetime
import sys
import pprint
import tfext.alexnet
import tfext.convnet
import tfext.fcconvnetv2
import tfext.utils
from helper import ALL_CATEGORIES
from common import run_training, get_first_model_path


def get_pathes(is_bbox_sq):
    assert not is_bbox_sq
    images_mat_pathes = {cat: '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/{}/images.mat'.format(
                        cat) for cat in ALL_CATEGORIES}

    output_dir = os.path.expanduser('/export/home/asanakoy/workspace/OlympicSports/cnn/fcconvnetv2_joint_categories_scratch')
    # output_dir = os.path.join(os.path.expanduser('~/tmp/tf_test'))
    mean_path = os.path.join(output_dir, 'mean.npy')
    return images_mat_pathes, mean_path, output_dir


def main(argv):
    images_mat_pathes, mean_path, output_dir = get_pathes(is_bbox_sq=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = {
        'im_shape': (227, 227, 3),
        'batch_size': 128,
        'fc_lr': 0.001,
        'conv_lr': 0.001,
        'fix_conv_iter': 1000000,
        'only_fc_train_op_iter': 1000000,

        'track_moving_averages': False,

        'test_layers': ['maxpool5', 'fc6', 'fc7'],
        'snapshot_path_to_restore': '/export/home/asanakoy/workspace/OlympicSports/cnn/fcconvnetv2_joint_categories_scratch/checkpoint-207500',
        'init_model': get_first_model_path(),
        'num_layers_to_init': 0,
        'network': tfext.fcconvnetv2.FcConvnetV2,

        'max_iter': 40000,
        'snapshot_iter': 2500,
        'test_step': 2500,
        'num_clustering_rounds': 200,
        'custom_params_clustering': {
            'num_initial_batches': 80,
        },

        'dataset': 'OlympicSports',
        'images_mat_pathes': images_mat_pathes,
        'mean_filepath': mean_path,
        'output_dir': output_dir,
        'seed': 1988,

        'random_shuffle_categories': True,
        'shuffle_every_epoch': True,
        'online_augmentations': True,
        'async_preload': False,
        'num_data_workers': 1,
        'gpu_memory_fraction': 0.40,
        'augmenter_params': dict(hflip=False, vflip=False,
                                 scale_to_percent=(1.0, 2 ** 0.5),
                                 scale_axis_equally=True,
                                 rotation_deg=10, shear_deg=4,
                                 translation_x_px=15, translation_y_px=15),

        'device_id': '/gpu:{}'.format(0)
    }
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(os.path.join(output_dir, 'train_params.dump_{}.txt'.format(suffix)), 'w') as f:
        f.write('{}\n'.format(pprint.pformat(params)))
    run_training(**params)


if __name__ == '__main__':
    main(sys.argv[1:])
