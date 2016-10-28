import h5py
import numpy as np
from os.path import join
from eval.image_getter import ImageGetterFromMat
import eval.features


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


def get_step_similarities(step, net, category, layers, **params):
    """
    Read similarities to compute clustering for different steps (rounds) of cnn training with reclustering.
    :param step:
    :param net:
    :param category:
    :param layers:
    :return:
    """
    # If step=0 read initial similarities otherwise compute similarities from the model
    if step == 0:
        data = h5py.File(params['pathtosim'], 'r')
        data2 = h5py.File(params['pathtosim_avg'], 'r')
        simMatrix = (data2['d'][()] + data2['d'][()].T) / 2.0
        flipMatrix = data['flipval'][()]
        return {'simMatrix': simMatrix, 'flipMatrix': flipMatrix}
    else:
        d, f = get_sim(net, category, layers)
        simMatrix_joined = np.dstack((d['simMatrix'], d['simMatrix_flip']))
        flipMatrix = simMatrix_joined.argmax(axis=2)
        simMatrix = simMatrix_joined.max(axis=2)
        return {'simMatrix': simMatrix, 'flipMatrix': flipMatrix}
