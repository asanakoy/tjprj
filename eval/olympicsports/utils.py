import h5py
import numpy as np
import os
from tqdm import tqdm
import PIL
import PIL.Image
from os.path import join

def get_num_classes(indices_path):
    mat_data = h5py.File(indices_path, 'r')
    num_cliques = int(np.array(mat_data['new_labels']).max() + 1)
    return num_cliques


def get_joint_categories_mean(output_path, image_shape=(227, 227)):
    """
    Compute save and return mean in HxWxC and RGB
    """
    image_shape = tuple(image_shape)
    if len(image_shape) != 2:
        raise ValueError

    if os.path.exists(output_path):
        return np.load(output_path)

    categories = ['basketball_layup',
                  'bowling',
                  'clean_and_jerk',
                  'discus_throw',
                  'diving_platform_10m',
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

    'Calculating mean for the images...'
    mean_img = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float64)
    for cat in categories:
        print 'Category {}'.format(cat)
        mat_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/{}/images.mat'.format(cat)
        dataset = h5py.File(mat_path, 'r')
        if 'images' in dataset:
            images = dataset['images'][...]  # matlab format CxWxH x N
        else:
            images = dataset['images_mat'][...]  # matlab format CxWxH x N
        images = images.transpose((0, 3, 2, 1))  # N x HxWxC matrix

        for i in tqdm(xrange(images.shape[0])):
            img = images[i, ...]
            if img.shape[:2] != image_shape:
                img = np.asarray(PIL.Image.fromarray(img).resize(image_shape, PIL.Image.ANTIALIAS))
            mean_img += img
            mean_img += np.fliplr(img)
    mean_img /= images.shape[0] * 2

    np.save(output_path, mean_img)
    return mean_img


def get_sim_pathes(model_name, iteration=0, round_id=0, **params):
    mean_str = '_nomean' if params['mean'] is None else ''

    if model_name == 'alexnet':
        iteration = 0
        round_id = None

    if round_id is None:
        init_model_path = join('/export/home/asanakoy/workspace/OlympicSports/cnn/',
                               model_name,
                               params['category'], 'checkpoint-{}'.format(iteration))

        sim_output_path = join('/export/home/asanakoy/workspace/OlympicSports/sim/tf/',
                               params['category'], 'simMatrix_{}_{}{}_iter_{}_{}_{}.mat'.
                               format(params['category'], model_name, mean_str, iteration,
                                      ''.join(params['layer_names']), params['norm_method']))
    else:
        init_model_path = join('/export/home/asanakoy/workspace/OlympicSports/cnn/',
                               model_name,
                               params['category'], 'checkpoint_r-{}'.format(round_id))

        sim_output_path = join('/export/home/asanakoy/workspace/OlympicSports/sim/tf/',
                               params['category'], 'simMatrix_{}_{}{}_rounds_{}_{}_{}.mat'.
                               format(params['category'], model_name, mean_str, round_id,
                                      ''.join(params['layer_names']), params['norm_method']))

    if model_name == 'alexnet':
        init_model_path = '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.tf'

    return init_model_path, sim_output_path
