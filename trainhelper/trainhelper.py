import h5py
import numpy as np
from os.path import join
from eval.image_getter import ImageGetterFromMat
import eval.features
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
    Calculate simMatrix and simMatrix_flip from existing net for specified
      OlympicSports category.
    Args can contain:
      mat_path,
      data_dir,
      batch_size,
      return_features
    Return: dict = {'simMatrix': sim_matrix, 'simMatrix_flip': sim_matrix_flip}

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
    return eval.features.compute_sim(net=net, **sim_params)


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
            simMatrix = np.load(pathtosim_avg)
            flipMatrix = np.zeros(simMatrix.shape)
        else:
            data = h5py.File(pathtosim, 'r')
            data2 = h5py.File(pathtosim_avg, 'r')
            simMatrix = (data2['d'][()] + data2['d'][()].T) / 2.0
            flipMatrix = data['flipval'][()]
        return {'simMatrix': simMatrix, 'flipMatrix': flipMatrix}
    else:
        if dataset.startswith('Caltech'):
            pars = {
            'mat_path': '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/crops/{}/images_mat.mat'.format(dataset,category),
            'data_dir': join(
                '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/{}_batch_128_10trans_shuffleMB1shuffleALL_0/mat/'.format(dataset)),
            }
            d = get_sim(net, category, layers, return_features=False, **pars)
        else:
            # TODO: if we use crops bbox_sq than change the pathes to mat_file and mean here
            d = get_sim(net, category, layers, return_features=False)
        simMatrix_joined = np.dstack((d['simMatrix'], d['simMatrix_flip']))
        flipMatrix = simMatrix_joined.argmax(axis=2)
        simMatrix = simMatrix_joined.max(axis=2)
        return {'simMatrix': simMatrix, 'flipMatrix': flipMatrix}


def get_params_clustering(dataset, category):
    """
    Params for clustering
    :param dataset:
    :param category:
    :return:
    """
    if dataset == 'OlympicSports':
        pathtosim = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/compute_similarities/' \
                    'sim_matrices/hog-lda/simMatrix_{}.mat'.format(category)
        pathtosim_avg = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/' \
                        'similarities_lda/d_{}.mat'.format(dataset, category)
        pathtoimg = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/image_data/' \
                    'imagePaths_{}.txt'.format(dataset, category)
        pathtocrops = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets' \
                      '/{}/crops/{}'.format(dataset, category)
        pathtoanchors = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/labels_HIWIs' \
                        '/processed_labels/anchors_{}.mat'.format(dataset, category)
        anchors = h5py.File(pathtoanchors, 'r')

        with open(pathtoimg) as f:
            imnames = f.readlines()
        seqnames = [n[2:25] for n in imnames]

        params = {
            'pathtosim': pathtosim,
            'pathtosim_avg': pathtosim_avg,
            'seqNames': seqnames,
            'imagePath': imnames,
            'pathToFolder': pathtocrops,
            'init_nCliques': 10,
            'init_nbatches': 100,
            'max_cliques_per_batch': 8,
            'batch_size': 128,
            'nSamples': 8,
            'anchors': anchors,
            'sampled_nbatches': 1000,
            'dataset': dataset,
            'category': category,
        }
    else:
        pathtosim_avg = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/sim/simMatrix_INIT.npy'\
            .format(dataset)
        pathtoimg = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/image_paths.txt'.format(dataset)
        pathtocrops = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/crops/{}'.format(dataset,
                                                                                                             category)
        with open(pathtoimg) as f:
            imnames = f.readlines()


        params = {
            'pathtosim': None,
            'pathtosim_avg': pathtosim_avg,
            'seqNames': None,
            'imagePath': imnames,
            'pathToFolder': pathtocrops,
            'init_nCliques': 5,
            'init_nbatches': 20,
            'max_cliques_per_batch': 8,
            'batch_size': 64,
            'nSamples': 8,
            'anchors': None,
            'sampled_nbatches': 1000,
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
    if params_clustering['clustering_round'] == 0:
        generator = BatchGenerator(**params_clustering)
        init_batches = generator.generateBatches(params_clustering['init_nbatches'])
        params_clustering['batches'] = init_batches
        params_clustering['sampler'] = BatchSampler(**params_clustering)
        params_clustering['sampler'].updateCliqueSampleProb(
            np.ones(len(params_clustering['sampler'].cliques)))
    else:
        params_clustering['sampler'].updateSimMatrix(params_clustering['simMatrix'])
        params_clustering['sampler'].transitiveCliqueComputation()

    # # Save batchsampler
    sampler_file = open(os.path.join(params_clustering['output_dir'], 'sampler_round_' + str(params_clustering['clustering_round']) + '.pkl'), 'wb')
    pickle.dump(params_clustering['sampler'].cliques, sampler_file, pickle.HIGHEST_PROTOCOL)
    sampler_file.close()

    indices = np.empty(0, dtype=np.int64)
    flipped = np.empty(0, dtype=np.bool)
    label = np.empty(0, dtype=np.int64)
    print 'Sampling batches'
    for i in tqdm(range(params_clustering['sampled_nbatches'])):
        batch = params_clustering['sampler'].sampleBatch(params_clustering['batch_size'],
                                                         params_clustering['max_cliques_per_batch'],
                                                         mode='random')
        _x, _f, _y = params_clustering['sampler'].parse_to_list(batch)
        indices = np.append(indices, _x.astype(dtype=np.int64))
        flipped = np.append(flipped, _f.astype(dtype=np.bool))
        label = np.append(label, _y.astype(dtype=np.int64))

    assert indices.shape[0] == flipped.shape[0] == label.shape[0], "Corrupted arguments for batch loader"
    return {'idxs': indices, 'flipvals': flipped, 'labels': label}, params_clustering
