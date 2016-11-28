# Copyright (c) 2016 Artsiom Sanakoyeu
import os.path
import time
import datetime
import sys
import pprint
import tfext.alexnet
import tfext.convnet
import tfext.fcconvnetv2
import tfext.utils
import demo.joint_olympicsports.helper
from joblib import Parallel, delayed
from demo.joint_olympicsports.common import run_training, get_first_model_path


def get_pathes(category):
    images_mat_pathes = {
        cat: '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/{}/images.mat'.format(
            cat) for cat in [category]}

    output_dir = '/export/home/asanakoy/workspace/OlympicSports/cnn/ft_alexnet_joint_categories_imagenet_softxe/{}'.format(
        category)
    output_dir = os.path.join(os.path.expanduser('~/tmp/tf_test'))
    # mean_path = os.path.join(output_dir, 'mean.npy')
    mean_path = '/export/home/asanakoy/workspace/OlympicSports/cnn/alexnet_joint_categories/mean.npy'
    return images_mat_pathes, mean_path, output_dir


def main(category):
    if category is None:
        category = 'vault'

    print category
    images_mat_pathes, mean_path, output_dir = get_pathes(category)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = {
        'im_shape': (227, 227, 3),
        'batch_size': 128,
        'fc_lr': 0.001,
        'conv_lr': 0.0001,
        'fix_conv_iter': 10000,
        'only_fc_train_op_iter': 10000,
        'fix_up_to_the_last': 0,

        'loss_type': 'soft_xe',

        'track_moving_averages': False,

        'test_layers': ['maxpool5', 'fc6', 'fc7'],
        'snapshot_path_to_restore': '/export/home/asanakoy/workspace/OlympicSports/cnn/alexnet_joint_categories/checkpoint-445004',
        'init_model': get_first_model_path(),
        'num_layers_to_init': 0,
        'network': tfext.alexnet.Alexnet,

        'max_iter': 10000,
        'snapshot_iter': 'save_the_best',
        'test_step': 200,
        'num_clustering_rounds': 1,
        'custom_params_clustering': {
            'num_initial_batches': 150,
        },

        'dataset': 'OlympicSports',
        'images_mat_pathes': images_mat_pathes,
        'mean_filepath': mean_path,
        'output_dir': output_dir,
        'categories': [category],
        'seed': 1988,

        'random_shuffle_categories': False,
        'shuffle_every_epoch': False,
        'online_augmentations': True,
        'async_preload': False,
        'num_data_workers': 1,
        'gpu_memory_fraction': 0.30,
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
    main(None)
    # ALL_CATEGORIES = [
    #     'basketball_layup',
    #     'bowling',
    #     'clean_and_jerk',
    #     'discus_throw',
    #     'diving_platform_10m',
    #     'diving_springboard_3m',
    #     'hammer_throw',
    #     'high_jump',
    #     'javelin_throw',
    #     'long_jump',
    #     'pole_vault',
    #     'shot_put',
    #     'snatch',
    #     'tennis_serve',
    #     'triple_jump',
    #     'vault']
    #
    # n_jobs = int(sys.argv[1])
    # print 'Running {} workers'.format(n_jobs)
    # Parallel(n_jobs=n_jobs)(
    #     delayed(main)(cat) for cat in ALL_CATEGORIES)
