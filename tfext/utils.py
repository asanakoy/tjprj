# Artsiom Sanakoyeu, 2016
from __future__ import division
import numpy as np
from eval.olympicsports.utils import get_sim
import h5py
from clustering.batchgenerator import BatchGenerator
from clustering.batchsampler import BatchSampler
from clustering.clique import Clique
import cPickle as pickle
import os

def fill_feed_dict(net, batch_loader, batch_size=128, phase='test'):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      batch_loader: BatchLoader, that provides batches of the data
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    assert phase in ['train', 'test']
    keep_prob = 1.0 if phase == 'test' else 0.5

    images_feed, labels_feed = batch_loader.get_next_batch(batch_size)
    # TODO: remove after I fix BatchLoader. Presently BatchLoader outputs CxHxW images.
    images_feed = images_feed.transpose((0, 2, 3, 1))

    feed_dict = {
        net.x: images_feed,
        net.y_gt: labels_feed,
        net.fc6_keep_prob: keep_prob,
        net.fc7_keep_prob: keep_prob
    }
    return feed_dict

def fill_feed_dict_magnet(net, mu_ph, unique_mu_ph, sigma_ph, batch_loader, centroider, batch_size=128, phase='test'):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      batch_loader: BatchLoader, that provides batches of the data
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    assert phase in ['train', 'test']
    keep_prob = 1.0 if phase == 'test' else 0.5

    images_feed, labels_feed = batch_loader.get_next_batch(batch_size)
    mu_feed = centroider.get_nearest_mu(labels_feed)
    _, unique_labels = np.unique(labels_feed, return_index=True)
    unique_mu_feed = mu_feed[unique_labels]
    sigma_feed = centroider.get_sigma(labels_feed)
    # TODO: remove after I fix BatchLoader. Presently BatchLoader outputs CxHxW images.
    images_feed = images_feed.transpose((0, 2, 3, 1))

    feed_dict = {
        net.x: images_feed,
        net.y_gt: labels_feed,
        mu_ph: mu_feed,
        unique_mu_ph: unique_mu_feed,
        sigma_ph: sigma_feed,
        net.fc6_keep_prob: keep_prob,
        net.fc7_keep_prob: keep_prob
    }
    return feed_dict


def calc_acuracy(net,
                 sess,
                 eval_correct,
                 batch_loader,
                 batch_size=128,
                 num_images=None):
    """Runs one correct_classified_top1 against the full epoch of data.

    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      batch_loader: BatchLoader, that provides batches of the data
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    if num_images is None:
        steps_num = len(batch_loader.indexlist) // batch_size
    else:
        assert num_images <= len(batch_loader.indexlist)
        steps_num = num_images // batch_size


    # WARNING: be careful! We change the internal pointer of the BatchLoader.
    old_batch_loader_position = batch_loader._cur

    batch_loader._cur = 0
    num_examples = steps_num * batch_size
    for step in xrange(steps_num):
        feed_dict = fill_feed_dict(net,
                                   batch_loader,
                                   batch_size=batch_size,
                                   phase='test')
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    # restore the internal pointer
    batch_loader._cur = old_batch_loader_position

    accuracy = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d Accuracy @ 1: %0.04f' %

          (num_examples, true_count, accuracy))


def get_params_clustering(dataset, category):
    """
    Params for clustering
    :param dataset:
    :param category:
    :return:
    """

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
        'nSamples': 8,
        'anchors': anchors,
        'sampled_nbatches': 1000,
        'dataset': dataset,
        'category': category,
    }

    return params


def get_similarities(step, net, category, layers, params):
    """
    Read similarities to compute clustering
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


def runClustering(params_clustering, params):
    """
    Run clustering assignment procedure and return arrays for BatchLoader in a dict
    :param kwargs_generator: arguments for generator
    :param kwargs_sampler: arguments for sampler
    :return: Dict of arrays for BatchLoader
    """
    if params_clustering['clustering_round'] == 0:
        generator = BatchGenerator(**params_clustering)
        init_batches = generator.generateBatches(init_nbatches=100)
        params_clustering['batches'] = init_batches
        params_clustering['sampler'] = BatchSampler(**params_clustering)
        params_clustering['sampler'].updateCliqueSampleProb(np.ones(len(params_clustering['sampler'].cliques)))
    else:
        params_clustering['sampler'].updateSimMatrix(params_clustering['simMatrix'])
        params_clustering['sampler'].transitiveCliqueComputation()

    # # Save batchsampler
    # sampler_file = open(os.path.join(params['output_dir'], 'sampler_round_' + str(params['clustering_round']) + '.pkl'), 'wb')
    # pickle.dump(params_clustering['sampler'], sampler_file, pickle.HIGHEST_PROTOCOL)
    # sampler_file.close()

    indices = np.empty(0, dtype=np.int64)
    flipped = np.empty(0, dtype=np.bool)
    label = np.empty(0, dtype=np.int64)
    for i in range(params_clustering['sampled_nbatches']):
        print "Sampling batch {}".format(i)
        batch = params_clustering['sampler'].sampleBatch(batch_size=128, max_cliques_per_batch=8, mode='random')
        _x, _f, _y = params_clustering['sampler'].parse_to_list(batch)
        indices = np.append(indices, _x.astype(dtype=np.int64))
        flipped = np.append(flipped, _f.astype(dtype=np.bool))
        label = np.append(label, _y.astype(dtype=np.int64))

    assert indices.shape[0] == flipped.shape[0] == label.shape[0], "Corrupted arguments for batch loader"
    return {'idxs': indices, 'flipvals': flipped, 'labels': label}, params_clustering
