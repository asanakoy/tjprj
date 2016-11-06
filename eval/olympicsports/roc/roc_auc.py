import sys
from os.path import join
import numpy as np
import h5py
import scipy.interpolate
import scipy.io
import sklearn.metrics as sklm
import os
import glob

POS_LABEL = 1


def covert_labels_to_dict(f):
    """
    Compute dict containing all info about labels.
    Pos label is 1, neg label is 0.
    """
    d = dict()
    assert f['anchors'].ndim == 1
    n_anchors = f['anchors'].shape[0]
    assert f['pos_ids'].shape[0] == f['neg_ids'].shape[0] == n_anchors
    assert f['neg_flipvals'].shape[0] == f['pos_flipvals'].shape[0] == n_anchors

    d['anchors'] = np.asarray(f['anchors'], dtype=np.int32)
    d['ids'] = [None] * n_anchors
    d['labels'] = [None] * n_anchors
    d['flipvals'] = [None] * n_anchors
    for i in xrange(n_anchors):
        d['ids'][i] = np.hstack([np.asarray(f['pos_ids'][i], dtype=np.int32),
                                 np.asarray(f['neg_ids'][i], dtype=np.int32)])

        d['labels'][i] = np.hstack([np.ones(f['pos_ids'][i].shape, dtype=np.bool),
                                    np.zeros(f['neg_ids'][i].shape, dtype=np.bool)])
        assert np.sum(d['labels'][i]) == f['pos_ids'][i].shape[0]

        d['flipvals'][i] = np.hstack([np.asarray(f['pos_flipvals'][i], dtype=np.bool),
                                      np.asarray(f['neg_flipvals'][i], dtype=np.bool)])
        assert np.sum(d['flipvals'][i]) == np.sum(f['pos_flipvals'][i]) + np.sum(
            f['neg_flipvals'][i])
    return d


def get_pr_auc(labels, scores, pos_class=1):
    labels_gt = labels == pos_class
    precision, recall, _ = \
        sklm.precision_recall_curve(labels_gt,
                                    scores,
                                    pos_label=1)
    average_precision = sklm.auc(recall, precision)
    return average_precision


def get_roc_auc(labels, scores, pos_class=1):
    labels_gt = labels == pos_class
    fpr, tpr, _ = sklm.roc_curve(labels_gt, scores, pos_label=pos_class)
    roc_auc = sklm.auc(fpr, tpr, reorder=True)
    return roc_auc, fpr, tpr


def compute_interpolated_roc_auc(labels_dict, false_pos_rate_list, true_pos_rate_list):
    """
    Average results: interpolation for all anchors at 101 grid points.
    Then get joined ROC Curve by averaging interpolated values at grid points.
    Args:
        labels_dict: dict of labels
        false_pos_rate_list: i-th element is the list of false positive rates for the i-th anchor
        true_pos_rate_list: i-th element is the list of true positive rates for the i-th anchor

    Return: joined ROC AUC
    """
    if len(false_pos_rate_list) != len(true_pos_rate_list):
        raise ValueError('fpr and tpr lists must be of the same size')

    grid_x = np.linspace(0, 1, num=101, endpoint=True)
    assert len(grid_x) == 101
    # grid_x = np.unique(np.hstack(false_pos_rate_list))
    grid_y = np.zeros((len(labels_dict['anchors']), len(grid_x)))
    for i in xrange(len(labels_dict['anchors'])):
        func = scipy.interpolate.interp1d(false_pos_rate_list[i],
                                          true_pos_rate_list[i],
                                          kind='linear', bounds_error=True)
        grid_y[i][...] = func(grid_x)
    mean_y = np.mean(grid_y, axis=0)
    interp_roc_auc = sklm.auc(grid_x, mean_y, reorder=True)
    return interp_roc_auc


def compute_roc(d, sim):
    stacked_sim_matrix = np.stack([sim['simMatrix'], sim['simMatrix_flip']], axis=2)
    assert stacked_sim_matrix.ndim == 3 and stacked_sim_matrix.shape[2] == 2
    assert stacked_sim_matrix[0, 0, 0] > stacked_sim_matrix[0, 0, 1]

    roc_auc_list = list()
    false_pos_rate_list = list()
    true_pos_rate_list = list()
    for i, anchor_id in enumerate(d['anchors']):
        scores = [stacked_sim_matrix[anchor_id, frame_id, flipval] for frame_id, flipval in
                  zip(d['ids'][i], d['flipvals'][i].astype(int))]
        assert len(scores) == len(d['ids'][i])
        roc_auc, fpr, tpr = get_roc_auc(d['labels'][i], scores, pos_class=1)
        roc_auc_list.append(roc_auc)
        false_pos_rate_list.append(fpr)
        true_pos_rate_list.append(tpr)
    avg_roc_auc = np.mean(roc_auc_list)
    interp_roc_auc = compute_interpolated_roc_auc(d, false_pos_rate_list, true_pos_rate_list)
    return avg_roc_auc, interp_roc_auc, roc_auc_list


def compute_roc_auc_from_sim(argv, path_sim_matrix=None, is_quiet=False):
    if len(argv) == 0:
        category = 'long_jump'
    else:
        category = argv[0]

    iter_id = 20000
    dataset_root = '/export/home/asanakoy/workspace01/datasets/OlympicSports/'

    if path_sim_matrix is None:
        suffix = 'with_bn_fc7'
        path_sim_matrix = join(dataset_root, 'sim/tf/', suffix, category,
                               'simMatrix_{}_tf_0.1conv_1fc_{}iter_{}_fc7_zscores.mat'.format(
                                   category, suffix + '_' if len(suffix) else '', iter_id))
    if not is_quiet:
        print 'Sim matrix path:', path_sim_matrix
    sim = scipy.io.loadmat(path_sim_matrix)

    labels_path = join(dataset_root,
                       'dataset_labeling/labels_hdf5_19.02.16/labels_{}.hdf5'.format(
                           category))
    with h5py.File(labels_path, mode='r') as f:
        d = covert_labels_to_dict(f)

    avg_roc_auc, interp_roc_auc, roc_auc_list = compute_roc(d, sim)
    print '{} n_acnhors: {} mean_ROC_AUC: {:.3f} interp_ROC_AUC: {:.3f}'.format(category, len(
        d['anchors']), avg_roc_auc, interp_roc_auc)
    return interp_roc_auc


def run_all_cat():
    categories = [
        'bowling',
        'long_jump',
        'basketball_layup',
        'clean_and_jerk',
        'discus_throw',
        'diving_platform_10m',
        'diving_springboard_3m',
        'hammer_throw',
        'high_jump',
        'javelin_throw',
        'pole_vault',
        'shot_put',
        'snatch',
        'tennis_serve',
        'triple_jump',
        'vault']
    categories = sorted(categories)
    for cat in categories:
        try:
            path_sim_matrix = '/export/home/asanakoy/workspace/OlympicSports/sim/tf/{0}/simMatrix_{0}_aug_less_aggressive__rounds_1_fc7_None'.format(cat)
            compute_roc_auc_from_sim([cat], path_sim_matrix=path_sim_matrix, is_quiet=True)
        except IOError as e:
            # print e
            print cat


if __name__ == '__main__':
    # run_all_cat()
    # compute_roc_auc_from_sim(['long_jump'], path_sim_matrix='/export/home/mbautist/Desktop/long_jump/simMatrix_long_jump_tf_0.1conv_1fc_growingiter2_iter_20000_fc7_zscores.mat')
    compute_roc_auc_from_sim(['long_jump'], path_sim_matrix='/export/home/mbautist/Desktop/fixed_magnet.mat')
    # compute_roc_auc_from_sim(['long_jump'])