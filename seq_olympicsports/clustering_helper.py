import numpy as np
from trainhelper import trainhelper
from clustering.batchgenerator import BatchGenerator
from clustering.batchsampler import BatchSampler
import h5py
from tqdm import tqdm
import os
import pickle


def run_reclustering(clustering_round=None,

                     sim_matrix=None,
                     flipvals=None,
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
    generator = BatchGenerator(sim_matrix=sim_matrix,
                               flipvals=flipvals,
                               seq_names=seq_names,
                               relative_image_pathes=relative_image_pathes,
                               crops_dir=crops_dir,
                               num_cliques_per_initial_batch=num_cliques_per_initial_batch,
                               num_samples_per_clique=num_samples_per_clique,
                               anchors=anchors,
                               seed=seed)
    init_batches = generator.generate_batches(num_initial_batches)
    sampler = BatchSampler(batches=init_batches,
                           sim_matrix=sim_matrix,
                           flipvals=flipvals,
                           seq_names=seq_names,
                           crops_dir=crops_dir,
                           relative_image_pathes=relative_image_pathes,
                           seed=seed)

    # Save batchsampler
    sampler_file = open(os.path.join(output_dir, 'sampler_round_' + str(
        clustering_round) + '.pkl'), 'wb')
    pickle.dump(sampler.cliques, sampler_file, pickle.HIGHEST_PROTOCOL)
    sampler_file.close()

    indices = np.empty(0, dtype=np.int64)
    flipped = np.empty(0, dtype=np.bool)
    label = np.empty(0, dtype=np.int64)
    print 'Sampling batches'
    for i in tqdm(range(num_batches_to_sample)):
        # print "Sampling batch {}".format(i)
        batch = sampler.sample_batch(batch_size,
                                     max_cliques_per_batch,
                                     mode='random')
        _x, _f, _y = sampler.parse_to_list(batch)
        assert len(_x) == len(_f) == len(_y) == batch_size
        indices = np.append(indices, _x.astype(dtype=np.int64))
        flipped = np.append(flipped, _f.astype(dtype=np.bool))
        label = np.append(label, _y.astype(dtype=np.int64))

    assert indices.shape[0] == flipped.shape[0] == label.shape[0], \
        'Corrupted arguments for batch loader'
    return {'idxs': indices, 'flipvals': flipped, 'labels': label}


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


def cluster_category(net, category_name, seed, output_dir):
    print('Clustering %s' % category_name)
    params_clustering = trainhelper.get_default_params_clustering('OlympicSports',
                                                                  category_name,
                                                                  num_initial_batches=None)

    sims_dict = trainhelper.get_sim(net=net, category=category_name, layers=['fc7'], return_features=False)
    simMatrix_joined = np.dstack((sims_dict['sim_matrix'], sims_dict['simMatrix_flip']))
    flipvals = simMatrix_joined.argmax(axis=2)
    sim_matrix = simMatrix_joined.max(axis=2)

    # Run clustering and update corresponding param fields
    params_clustering['sim_matrix'] = sim_matrix
    params_clustering['flipvals'] = flipvals
    params_clustering['clustering_round'] = 0
    params_clustering['output_dir'] = output_dir
    # TODO: maybe recluster next round not on anchors
    index_dict, _ = run_reclustering(clustering_round=None,

                                     sim_matrix=sim_matrix,
                                     flipvals=flipvals,
                                     seq_names=params_clustering['seq_names'],
                                     crops_dir=params_clustering['crops_dir'],
                                     relative_image_pathes=params_clustering['relative_image_pathes'],
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

    index_dict['labels'] = np.asarray(index_dict['labels'])
    # num_classes = index_dict['labels'].max() + 1