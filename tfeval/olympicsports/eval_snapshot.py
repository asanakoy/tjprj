# Copyright (c) 2016 Artsiom Sanakoyeu
import os.path
import time
import datetime
import sys
import pprint
import numpy as np
import tfext
import tfext.alexnet
import tfext.convnet
import tfext.fcconvnetv2
import tfext.utils
import demo.joint_olympicsports.helper
from joblib import Parallel, delayed
from demo.joint_olympicsports.common import run_training, get_first_model_path
import tfeval.olympicsports.roc.roc_from_net
import tensorflow as tf


def eval_net(net, category, layer_names, mat_path, mean_path):
    roc_auc_dict = tfeval.olympicsports.roc. \
        roc_from_net.compute_roc_auc_from_net(net,
                                              category,
                                              layer_names,
                                              mat_path=mat_path,
                                              mean_path=mean_path,
                                              batch_size=256,
                                              norm_method=None)
    return roc_auc_dict


def eval_all_cat(net, categories, test_layers, images_mat_pathes, mean_filepath):
    aucs = {key: list() for key in test_layers}
    for category in categories:
        cur_cat_roc_dict = eval_net(net, category,
                                    test_layers,
                                    mat_path=images_mat_pathes[category],
                                    mean_path=mean_filepath)
        for layer_name, auc in cur_cat_roc_dict.iteritems():
            aucs[layer_name].append(auc)
    scores = dict()
    for layer_name, auc_list in aucs.iteritems():
        cur_layer_mean = np.mean(auc_list)
        scores[layer_name] = cur_layer_mean
    return scores


def run_eval(net, categories, test_layers, images_mat_pathes, mean_filepath, output_dir):

    best_scores = {key: 0.0 for key in test_layers}
    global_step = net.sess.run(net.global_iter_counter)
    scores = eval_all_cat(net, categories, test_layers, images_mat_pathes, mean_filepath)

    for layer_name, layer_auc in scores.iteritems():
            print '{}: {}'.format(layer_name, layer_auc)
            best_scores[layer_name] = layer_auc
            open(os.path.join(output_dir,
                              '{}_{:.3f}_iter-{}'.format(layer_name, layer_auc,
                                                         global_step)),
                 mode='w').close()


def setup_network(network_class, snapshot_path_to_restore, num_layers_to_init, gpu_memory_fraction):
    net = network_class(gpu_memory_fraction=gpu_memory_fraction)
    net.sess.run(tf.global_variables_initializer())
    if snapshot_path_to_restore is not None:
        print('Restoring 7 layers from snapshot {}'.format(snapshot_path_to_restore))
        net.restore_from_snapshot(snapshot_path_to_restore, num_layers_to_init, restore_iter_counter=False)
    return net


def get_pathes(categories):
    images_mat_pathes = {
    cat: '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/{}/images.mat'.format(
        cat) for cat in categories}

    subfolder = ''
    if len(categories) == 1:
        subfolder = categories[0]
    output_dir = os.path.join(
        '/export/home/asanakoy/workspace/OlympicSports/cnn/sufflelearn_shufflelearnnet_eval', subfolder)
    # mean_path = os.path.join(output_dir, 'mean.npy')
    mean_path = os.path.join(
        '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' +
        categories[0] + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/mean.npy')
    # mean_path = None
    return images_mat_pathes, mean_path, output_dir


def main(categories):
    if categories is None:
        categories = ['bowling']
    print categories
    images_mat_pathes, mean_path, output_dir = get_pathes(categories)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    network_class = tfext.Shufflelearnnet
    num_layers_to_init = 7
    snapshot_path_to_restore = '/export/home/asanakoy/workspace/tfprj/data/shuffle_learn/shuffle_learn.tf'
    gpu_memory_fraction = 0.3
    test_layers = ['maxpool5', 'fc6', 'fc7']

    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(os.path.join(output_dir, 'train_params.dump_{}.txt'.format(suffix)), 'w') as f:
        f.write('network_class: {}\nnum_layers_to_init: {}\n'
                'snapshot_path_to_restore: {}\n'.format(network_class,
                                                        num_layers_to_init,
                                                        snapshot_path_to_restore))

    with tf.Graph().as_default():
        net = setup_network(network_class,
                            snapshot_path_to_restore,
                            num_layers_to_init,
                            gpu_memory_fraction)
    run_eval(net, categories, test_layers, images_mat_pathes, mean_path, output_dir)


if __name__ == '__main__':
    ALL_CATEGORIES = [
        'diving_platform_10m',
        'basketball_layup',
        'bowling',
        'clean_and_jerk',
        'discus_throw',
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

    n_jobs = int(sys.argv[1])
    print 'Running {} workers'.format(n_jobs)
    Parallel(n_jobs=n_jobs)(
        delayed(main)([cat]) for cat in ALL_CATEGORIES)
