from os.path import join
import h5py
import numpy as np
from eval.image_getter import ImageGetterFromMat
import eval.features


def get_num_classes(indices_path):
    mat_data = h5py.File(indices_path, 'r')
    num_cliques = int(np.array(mat_data['new_labels']).max() + 1)
    return num_cliques


def get_sim(net, category, layer_names, **args):
    """
    Calculate simMatrix and simMatrix_flip from existing net for specified
        OlympicSports category.
    Args can contain:
        mat_path,
        data_dir
        batch_size
    Return: dict = {'simMatrix': sim_matrix, 'simMatrix_flip': sim_matrix_flip}

    Example: d = get_sim(net, 'long_jump', ['fc7'])
    """
    default_params = dict(
        mat_path='/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/' + category + '/images_test.mat',
        data_dir=join(
            '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' + category + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/'),
        batch_size=256
    )
    for key in args.keys():
        if key not in default_params:
            raise Exception('Not expected argument: {}'.format(key))
    default_params.update(args)

    mean_path = join(default_params['data_dir'], 'mean.npy')
    sim_params = {
        'category': category,
        'layer_names': layer_names,
        'mean': np.load(mean_path),
        'batch_size': default_params['batch_size'],
        'im_shape': (227, 227),
        'image_getter': ImageGetterFromMat(default_params['mat_path']),
        'return_features': True
    }
    return eval.features.compute_sim(net=net, **sim_params)