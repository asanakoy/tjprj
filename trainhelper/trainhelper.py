import h5py
import numpy as np
from os.path import join
from tfeval.image_getter import ImageGetterFromMat
import tfeval.features
import h5py
from clustering.batchgenerator import BatchGenerator
from clustering.batchsampler import BatchSampler
from clustering.clique import Clique
import cPickle as pickle
import os
from tqdm import tqdm


def get_alexnet_snapshot_path():
    return '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.tf'


def get_sim(net, category, layer_names, **args):
    """
    Calculate sim_matrix and simMatrix_flip from existing net for specified
      OlympicSports category.
    Args can contain:
      mat_path,
      data_dir,
      batch_size,
      return_features
    Return: dict = {'sim_matrix': sim_matrix, 'simMatrix_flip': sim_matrix_flip}

    Example: d = get_sim(net, 'long_jump', ['fc7'], batch_size=128, return_features=False)
    """
    default_params = dict(
        mat_path='/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/' + category + '/images_test.mat',
        data_dir=join(
            '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' + category + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/'),
        batch_size=256,
        return_features=False
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
        'return_features': default_params['return_features']
    }
    return tfeval.features.compute_sim(net=net, **sim_params)


def get_step_similarities(step, net, category, dataset, layers, pathtosim=None, pathtosim_avg=None):
    """
    Read similarities to compute clustering for different steps (rounds) of cnn training with reclustering.
    Args:
    step: Round of the CNN training.
    pathtosim, pathtosim_avg: path to initial sims to load on step 0 # TODO: Miguel?
    :return:
    """
    # If step=0 read initial similarities otherwise compute similarities from the model
    if step == 0:
        if pathtosim_avg is None:
            raise ValueError
        if dataset.startswith('Caltech'):
            sim_matrix = np.load(pathtosim_avg)
            flipvals = np.zeros(sim_matrix.shape)
        elif dataset.startswith('pascal'):
            sim_matrix = np.load(pathtosim_avg)
            flipvals = np.zeros(sim_matrix.shape)
        else:
            data = h5py.File(pathtosim, 'r')
            # TODO: ask Miguel what is matrix d??? Look like this is not max amd not an average
            avg_sim = h5py.File(pathtosim_avg, 'r')
            sim_matrix = (avg_sim['d'][()] + avg_sim['d'][()].T) / 2.0
            flipvals = data['flipval'][()]
            # simMAatrix: merged similarity matrix, containing max
            # flipvals is just flipvals
        return {'sim_matrix': sim_matrix, 'flipvals': flipvals}
    else:
        if dataset.startswith('Caltech'):
            pars = {
            'mat_path': '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/crops/{}/images_mat.mat'.format(dataset,category),
            'data_dir': join(
                '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/{}_batch_128_10trans_shuffleMB1shuffleALL_0/mat/'.format(dataset)),
            }
            d = get_sim(net, category, layers, return_features=False, **pars)

        if dataset.startswith('pascal'):
            pars = {
            'mat_path': '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/images_CVPR17/images.mat',
            'data_dir': join(
                '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/{}_batch_128_10trans_shuffleMB1shuffleALL_0/mat/'.format(dataset)),
            }
            d = get_sim(net, category, layers, return_features=False, **pars)

        else:
            # TODO: if we use crops bbox_sq than change the pathes to mat_file and mean here
            d = get_sim(net, category, layers, return_features=False)
        simMatrix_joined = np.dstack((d['simMatrix'], d['simMatrix_flip']))
        flipvals = simMatrix_joined.argmax(axis=2)
        sim_matrix = simMatrix_joined.max(axis=2)
        return {'sim_matrix': sim_matrix, 'flipvals': flipvals}


def get_default_params_clustering(dataset, category):
    """
    Params for clustering
    :param dataset:
    :param category:
    :return:
    """
    if dataset == 'OlympicSports':
        sim_path = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/compute_similarities/' \
                    'sim_matrices/hog-lda/simMatrix_{}.mat'.format(category)
        avgsim_path = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/' \
                        'similarities_lda/d_{}.mat'.format(dataset, category)
        imagepathes_file = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/image_data/' \
                    'imagePaths_{}.txt'.format(dataset, category)
        crops_dir = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets' \
                      '/{}/crops/{}'.format(dataset, category)
        anchors_path = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/labels_HIWIs' \
                        '/processed_labels/anchors_{}.mat'.format(dataset, category)
        anchors = h5py.File(anchors_path, 'r')

        with open(imagepathes_file) as f:
            imnames = f.readlines()
        seqnames = [n[2:25] for n in imnames]

        params = {
            'pathtosim': sim_path,
            'pathtosim_avg': avgsim_path,
            'seq_names': seqnames,
            'relative_image_pathes': imnames,
            'crops_dir': crops_dir,
            'num_cliques_per_initial_batch': 10,
            'num_initial_batches': 100,
            'max_cliques_per_batch': 8,
            'batch_size': 128,
            'num_samples_per_clique': 8,
            'anchors': anchors,
            'num_batches_to_sample': 1000,
            'dataset': dataset,
            'category': category,
        }
    elif dataset.startswith('Caltech'):
        avgsim_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/sim/simMatrix_INIT.npy'\
            .format(dataset)
        imagepathes_file = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/image_paths.txt'.format(dataset)
        crops_dir = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/crops/{}'.format(dataset,
                                                                                                             category)
        with open(imagepathes_file) as f:
            imnames = f.readlines()


        params = {
            'pathtosim': None,
            'pathtosim_avg': avgsim_path,
            'seq_names': None,
            'relative_image_pathes': imnames,
            'crops_dir': crops_dir,
            'num_cliques_per_initial_batch': 10,
            'num_initial_batches': 100,
            'max_cliques_per_batch': 8,
            'batch_size': 128,
            'num_samples_per_clique': 8,
            'anchors': None,
            'num_batches_to_sample': 1000,
            'dataset': dataset,
            'category': category,
        }
    elif dataset.startswith('STL'):
        params = {
            'pathtosim': None,
            'pathtosim_avg': None,
            'seq_names': None,
            'relative_image_pathes': None,
            'crops_dir': None,
            'num_cliques_per_initial_batch': 10,
            'num_initial_batches': 100,
            'max_cliques_per_batch': 8,
            'batch_size': 128,
            'num_samples_per_clique': 8,
            'anchors': None,
            'num_batches_to_sample': 10000,
            'dataset': dataset,
            'category': category,
        }
    elif dataset.startswith('pascal'):
        imagepathes_file = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/path_images.txt'
        crops_dir = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/crops_227x227/'
        with open(imagepathes_file) as f:
            imnames = f.readlines()

        params = {
            'pathtosim': None,
            'pathtosim_avg': '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/sim/simMatrix_stackedpca_train.npy',
            'seq_names': None,
            'relative_image_pathes': imnames,
            'crops_dir': crops_dir,
            'num_cliques_per_initial_batch': 10,
            'num_initial_batches': 100,
            'max_cliques_per_batch': 8,
            'batch_size': 128,
            'num_samples_per_clique': 8,
            'anchors': None,
            'num_batches_to_sample': 100,
            'dataset': dataset,
            'category': category,
        }

    return params


def runClustering(**params_clustering):
    """
    Run clustering assignment procedure and return arrays for BatchLoader in a dict
    :param kwargs_generator: arguments for generator
    :param kwargs_sampler: arguments for sampler
    :return: Dict of arrays for BatchLoader
    """
    generator = BatchGenerator(**params_clustering)
    init_batches = generator.generate_batches(params_clustering['num_initial_batches'])
    sampler = BatchSampler(batches=init_batches, **params_clustering)
    sampler.set_clique_sample_prob(
        np.ones(len(sampler.cliques)))
    if params_clustering['clustering_round'] > 0:
        # Grow cliques
        print 'Growing cliques'
        sampler.set_sim_matrix(params_clustering['sim_matrix'])
        sampler.transitive_clique_computation()

    # # Save batchsampler
    sampler_file = open(os.path.join(params_clustering['output_dir'], 'sampler_round_' + str(params_clustering['clustering_round']) + '.pkl'), 'wb')
    pickle.dump(sampler.cliques, sampler_file, pickle.HIGHEST_PROTOCOL)
    sampler_file.close()

    indices = np.empty(0, dtype=np.int64)
    flipped = np.empty(0, dtype=np.bool)
    label = np.empty(0, dtype=np.int64)
    print 'Sampling batches'
    for i in tqdm(range(params_clustering['num_batches_to_sample'])):
        batch = sampler.sample_batch(params_clustering['batch_size'],
                                     params_clustering['max_cliques_per_batch'],
                                     mode='random')
        _x, _f, _y = sampler.parse_to_list(batch)
        indices = np.append(indices, _x.astype(dtype=np.int64))
        flipped = np.append(flipped, _f.astype(dtype=np.bool))
        label = np.append(label, _y.astype(dtype=np.int64))

    assert indices.shape[0] == flipped.shape[0] == label.shape[0], "Corrupted arguments for batch loader"
    return {'idxs': indices, 'flipvals': flipped, 'labels': label}, params_clustering


def runClusteringSTL(**params_clustering):
    """

    :param params_clustering:
    :return:
    """
    # Load cliques from list and initilize batch loader
    pathtocliquesfile = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/STL10/groups/cliques_stackedfeatures.pkl'
    cliques_file = open(pathtocliquesfile, 'rb')
    cliques = pickle.load(cliques_file)
    sampler = BatchSampler(**{'batches': [cliques[0:params_clustering['num_classes']]]})
    sampler.set_clique_sample_prob(np.random.rand(params_clustering['num_classes']))
    indices = np.empty(0, dtype=np.int64)
    flipped = np.empty(0, dtype=np.bool)
    label = np.empty(0, dtype=np.int64)
    print 'Sampling batches'
    for i in tqdm(range(params_clustering['num_batches_to_sample'])):
        batch = sampler.sample_batch(params_clustering['batch_size'], params_clustering['max_cliques_per_batch'],
                                     mode='random')
        _x, _f, _y = sampler.parse_to_list(batch)
        indices = np.append(indices, _x.astype(dtype=np.int64))
        flipped = np.append(flipped, _f.astype(dtype=np.bool))
        label = np.append(label, _y.astype(dtype=np.int64))

    assert indices.shape[0] == flipped.shape[0] == label.shape[0], "Corrupted arguments for batch loader"
    return {'idxs': indices, 'flipvals': flipped, 'labels': label}, params_clustering
