from os.path import join
import numpy as np
import h5py
import scipy.stats.mstats as stat
import scipy.spatial.distance as spdis
import sklearn
import eval.image_getter
import eval.features
from eval.olympicsports.roc.roc_auc import covert_labels_to_dict, \
    compute_interpolated_roc_auc, get_roc_auc


def __get_similarity_score(feats, frame_id_a, frame_id_b, flipval):
    u = feats['features'][frame_id_a, ...]
    v = feats['features' if not flipval else 'features_flipped'][frame_id_b, ...]
    return 2 - spdis.correlation(u, v)


def compute_roc_auc_from_net(net, category, layer_names,
                             mat_path=None, mean_path=None,
                             batch_size=256,
                             norm_method=None,
                             load_all_in_memory=True):
    """Calculate ROC AUC on the current iteration of the net
    Args:
      net: network
      load_all_in_memory: load all images from the category in memory
    Return: ROC AUC for the category
    """
    if mat_path is None:
        mat_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/' + category + '/images_test.mat'
    if mean_path is None:
        mean_path = join(
            '/export/home/mbautist/Desktop/workspace/cnn_similarities/data/mat_files/cliqueCNN/' + category + '_batch_128_10trans_shuffleMB1shuffleALL_0/mat/mean.npy')
    labels_path = join(
        '/export/home/asanakoy/workspace/OlympicSports/dataset_labeling/labels_hdf5_19.02.16/labels_{}.hdf5'.format(
            category))
    with h5py.File(labels_path, mode='r') as f:
        labels_dict = covert_labels_to_dict(f)

    used_frame_ids = list(labels_dict['anchors'])
    for ids in labels_dict['ids']:
        used_frame_ids.extend(ids)
    used_frame_ids = np.unique(used_frame_ids)
    pos_lookup = {frame_id: pos for pos, frame_id in enumerate(used_frame_ids)}

    # Extracting features
    accepable_methods = [None, 'zscores', 'unit_norm']
    if norm_method not in accepable_methods:
        raise ValueError('unknown norm method: {}. Use one of {}'.format(norm_method,
                                                                         accepable_methods))
    feats_params = {
        'category': category,
        'layer_names': layer_names,
        'mean': np.load(mean_path),
        'batch_size': batch_size,
        'im_shape': (227, 227),
        'image_getter': eval.image_getter.ImageGetterFromMat(mat_path, load_all_in_memory=load_all_in_memory)
    }

    feats = dict()
    feats['features'] = eval.features.extract_features(False, net=net, frame_ids=used_frame_ids, **feats_params)
    feats['features_flipped'] = eval.features.extract_features(True, net=net, frame_ids=used_frame_ids, **feats_params)
    stacked_features = dict()
    for key, val in feats.iteritems():
        stacked_features[key] = np.hstack(val.values())
        if norm_method == 'zscores':
            stacked_features[key] = stat.zscore(stacked_features[key], axis=0)
        elif norm_method == 'unit_norm':
            # in-place
            sklearn.preprocessing.normalize(stacked_features[key], norm='l2', axis=1,
                                            copy=False)
    feats = stacked_features

    # Calculating ROC AUC
    false_pos_rate_list = list()
    true_pos_rate_list = list()
    for i, anchor_id in enumerate(labels_dict['anchors']):
        scores = [__get_similarity_score(feats, pos_lookup[anchor_id], pos_lookup[frame_id], flipval) for
                  frame_id, flipval in
                  zip(labels_dict['ids'][i], labels_dict['flipvals'][i].astype(int))]

        assert len(scores) == len(labels_dict['ids'][i])
        roc_auc, fpr, tpr = get_roc_auc(labels_dict['labels'][i], scores, pos_class=1)
        false_pos_rate_list.append(fpr)
        true_pos_rate_list.append(tpr)
    roc_auc = compute_interpolated_roc_auc(labels_dict, false_pos_rate_list,
                                           true_pos_rate_list)
    return roc_auc
