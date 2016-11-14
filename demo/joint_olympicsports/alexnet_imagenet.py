# Copyright (c) 2016 Artsiom Sanakoyeu
import os.path
import time
import sys
import pprint
import tfext.alexnet
import tfext.convnet
import tfext.utils
from helper import ALL_CATEGORIES
from common import run_training, get_first_model_path


def get_pathes(is_bbox_sq):
    assert not is_bbox_sq
    images_mat_pathes = {cat: '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/{}/images.mat'.format(
                        cat) for cat in ALL_CATEGORIES}

    output_dir = os.path.expanduser('/export/home/asanakoy/workspace/OlympicSports/cnn/alexnet_joint_categories')
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
        'conv_lr': 0.0001,
        'fix_conv_iter': 20000,
        'only_fc_train_op_iter': 20000,

        'test_layers': ['maxpool5', 'fc6', 'fc7', 'fc8'],
        'snapshot_path_to_restore': '/export/home/asanakoy/workspace/OlympicSports/cnn/alexnet_joint_categories/checkpoint-445004',
        'init_model': get_first_model_path(),
        'num_layers_to_init': 0,
        'network': tfext.alexnet.Alexnet,

        'max_iter': 25000,
        'snapshot_iter': 2500,
        'test_step': 2500,
        'num_clustering_rounds': 200,
        'init_nbatches': 110,

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
        'gpu_memory_fraction': 0.45,
        'augmenter_params': dict(hflip=False, vflip=False,
                                 scale_to_percent=(1.0, 2 ** 0.5),
                                 scale_axis_equally=True,
                                 rotation_deg=10, shear_deg=4,
                                 translation_x_px=15, translation_y_px=15),

        'device_id': '/gpu:{}'.format(0)
    }
    with open(os.path.join(output_dir, 'train_params.dump.txt'), 'w') as f:
        f.write('{}\n'.format(pprint.pformat(params)))
    run_training(**params)


if __name__ == '__main__':
    main(sys.argv[1:])
