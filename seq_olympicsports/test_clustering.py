import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf

import tfext
from trainhelper import trainhelper
from clustering.seq.clipbatchsampler import ClipBatchSampler
from clustering.seq.clipbatchgenerator import ClipBatchGenerator


def run_reclustering(clustering_round=None,

                     clip_len=None,
                     sim_matrix=None,
                     seq_names=None,
                     crops_dir=None,
                     relative_image_pathes=None,
                     num_initial_batches=None,
                     num_cliques_per_initial_batch=None,
                     num_samples_per_clique=None,
                     anchors=None,

                     batch_size=None,
                     num_batches_to_sample=None,
                     max_cliques_per_batch=None,
                     output_dir=None,
                     seed=None):
    """
    Run clustering assignment procedure and return arrays for BatchLoader in a dict
    Returns:
      Dict of arrays for BatchLoader
    """
    generator = ClipBatchGenerator(sim_matrix=sim_matrix,
                                   clip_len=clip_len,
                                   seq_names=seq_names,
                                   relative_image_pathes=relative_image_pathes,
                                   crops_dir=crops_dir,
                                   num_cliques_per_initial_batch=num_cliques_per_initial_batch,
                                   num_samples_per_clique=num_samples_per_clique,
                                   anchors=anchors,
                                   seed=seed)
    init_batches = generator.generate_batches(num_initial_batches)

    clips = list()
    flipvals = list()
    labels = list()
    print 'Sampling batches'
    for i, batch in tqdm(enumerate(init_batches)):
        # print "Sampling batch {}".format(i)
        _clips, _flipvals, _labels = ClipBatchSampler.parse_to_list(batch)
        clips.append(_clips)
        flipvals.append(flipvals)
        labels.append(labels)

    for batch in init_batches[:4]:
        for clique in batch:
            clique.visualize()
    plt.show()

    clips = np.vstack(clips)
    flipvals = np.hstack(flipvals)
    labels = np.hstack(labels)
    assert flipvals.shape == labels.shape, 'Corrupted arguments for batch loader'
    assert clips.shape[0] == len(flipvals) == len(labels)
    return clips, flipvals, labels


def get_params_clustering(dataset, category, num_initial_batches=None):
    """
    Params for clustering
    """
    assert dataset == 'OlympicSports'
    pathtosim = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/compute_similarities/' \
                'sim_matrices/hog-lda/simMatrix_{}.mat'.format(category)
    pathtosim_avg = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/' \
                    'similarities_lda/d_{}.mat'.format(dataset, category)
    img_pathes_file = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/image_data/' \
                      'imagePaths_{}.txt'.format(dataset, category)
    pathtocrops = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets' \
                  '/{}/crops/{}'.format(dataset, category)
    pathtoanchors = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/labels_HIWIs' \
                    '/processed_labels/anchors_{}.mat'.format(dataset, category)
    anchors = h5py.File(pathtoanchors, 'r')

    with open(img_pathes_file) as f:
        relative_img_pathes = f.readlines()
    seqnames = [path[2:25] for path in relative_img_pathes]

    params = {
        'pathtosim': pathtosim,
        'pathtosim_avg': pathtosim_avg,
        'seq_names': seqnames,
        'relative_image_pathes': relative_img_pathes,
        'crops_dir': pathtocrops,
        'num_cliques_per_initial_batch': 10,
        'num_initial_batches': num_initial_batches,
        'max_cliques_per_batch': 8,
        'batch_size': 128,
        'num_samples_per_clique': 8,
        'anchors': anchors,
        'num_batches_to_sample': 1000,
        'dataset': dataset,
        'category': category,
    }

    if params['num_initial_batches'] is None:
        print 'Building batches on anchors'
        params['num_initial_batches'] = len(params['anchors']['anchor'])

    return params


def cluster_category(net, category_name, layers=None, output_dir=None, seed=None):
    print('Clustering %s' % category_name)
    params_clustering = trainhelper.get_default_params_clustering('OlympicSports',
                                                                  category_name)

    sims_dict = trainhelper.get_sim(net=net, category=category_name, layer_names=layers, return_features=False)
    sim_matrix_joined = np.dstack((sims_dict['sim_matrix'], sims_dict['simMatrix_flip']))

    clips, flipvals, labels = run_reclustering(clustering_round=None,

                                               clip_len=5,
                                               sim_matrix=sim_matrix_joined,
                                               seq_names=params_clustering['seq_names'],
                                               crops_dir=params_clustering['crops_dir'],
                                               relative_image_pathes=params_clustering[
                                                   'relative_image_pathes'],
                                               num_initial_batches=100,
                                               num_cliques_per_initial_batch=10,
                                               num_samples_per_clique=8,
                                               anchors=params_clustering['anchors'],

                                               batch_size=128,
                                               num_batches_to_sample=1000,
                                               max_cliques_per_batch=8,
                                               output_dir=output_dir,
                                               seed=seed
                                                 )


def setup_network(network_class, snapshot_path_to_restore, num_layers_to_init,
                  gpu_memory_fraction):
    net = network_class(gpu_memory_fraction=gpu_memory_fraction)
    net.sess.run(tf.global_variables_initializer())
    if snapshot_path_to_restore is not None:
        print('Restoring 7 layers from snapshot {}'.format(snapshot_path_to_restore))
        net.restore_from_snapshot(snapshot_path_to_restore, num_layers_to_init,
                                  restore_iter_counter=False)
    return net

if __name__ == '__main__':
    output_dir = '/export/home/asanakoy/tmp/clip_cliques'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    net = setup_network(tfext.Alexnet, '/export/home/asanakoy/workspace/OlympicSports/cnn/alexnet_joint_categories/checkpoint-465004', 7, 0.5)
    cluster_category(net, 'pole_vault', layers=['fc7'], seed=1993, output_dir=output_dir)
