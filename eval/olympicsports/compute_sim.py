# Original work Copyright 2015 The TensorFlow Authors. Licensed under the Apache License v2.0 http://www.apache.org/licenses/LICENSE-2.0
# Modified work Copyright (c) 2016 Artsiom Sanakoyeu

# pylint: disable=missing-docstring
import os.path
from os.path import join
import time
import sys
import numpy as np
from eval.image_getter import ImageGetterFromMat
import eval.features
import eval.olympicsports.utils


def main(argv):
    if len(argv) > 1:
        category = argv[1]
    else:
        category = 'tennis_serve'
    model_name = 'tf_0.1conv_1fc'
    mat_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/' + category + '/images_test.mat'

    data_dir = join(
        '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' + category + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/')
    train_indices_path = join(data_dir,
                              category + '_batch_128_10trans_shuffleMB1shuffleALL_0_train.mat')
    mean_path = join(data_dir, 'mean.npy')
    mean = np.load(mean_path)
    num_classes = eval.olympicsports.utils.get_num_classes(train_indices_path)

    iteration = 20000
    init_model = '/export/home/asanakoy/workspace01/datasets/OlympicSports/cnn/{}/checkpoint-{}'. \
        format(category, iteration)

    params = {
        'category': category,
        'model_name': model_name,
        'iter': iteration,
        'layer_names': ['fc7'],
        'image_getter': ImageGetterFromMat(mat_path),
        'mean': mean,
        'im_shape': (227, 227),
        'batch_size': 256,
        'num_classes': num_classes,
        'snapshot_path': init_model,

        'sim_output_dir': join(
            '/export/home/asanakoy/workspace01/datasets/OlympicSports/sim/tf', category),
        'device_id': '/gpu:{}'.format(int(argv[0]))
    }
    eval.features.compute_sim_and_save(**params)


if __name__ == '__main__':
    main(sys.argv[1:])
