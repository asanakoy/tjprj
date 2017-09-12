import sys
from os.path import join
import numpy as np
import h5py
import scipy.interpolate
import scipy.io
import hdf5storage
import deepdish.io as dio
import sklearn.metrics as sklm
import os
import glob
from tqdm import tqdm

from roc_auc import get_roc_auc, covert_labels_to_dict, compute_interpolated_roc_auc


def read_seq_names(dataset, category):
    img_pathes_file = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/{}/image_data/' \
                      'imagePaths_{}.txt'.format(dataset, category)
    with open(img_pathes_file) as f:
        relative_img_pathes = f.readlines()
    seqnames = [path[2:25] for path in relative_img_pathes]
    return seqnames


def compute_similarity(stacked_sim_matrix, seq_names, frame_id_1, frame_id_2, flipval):
    MAX_CLIP_LEN = 5
    ids = [frame_id_1, frame_id_2]
    clips = list()
    clip_len = MAX_CLIP_LEN
    for frame_id in ids:
        cur_clip_len = 0
        while seq_names[frame_id - cur_clip_len] == seq_names[frame_id]:
            cur_clip_len += 1
        clip_len = min(clip_len, cur_clip_len)

    # if clip_len < MAX_CLIP_LEN:
    #     print '{}, {}: clip_len={}'.format(frame_id_1, frame_id_2, clip_len)

    for frame_id in ids:
        clips.append(range(frame_id - clip_len + 1, frame_id + 1))
    sim_values = list()
    for i, j in zip(*clips):
        sim_values.append(stacked_sim_matrix[i, j, flipval])
    return np.mean(sim_values)


def compute_similarity_middle(stacked_sim_matrix, seq_names, frame_id_1, frame_id_2, flipval,
                              max_clip_radius=4):
    """

    Args:
        stacked_sim_matrix:
        seq_names:
        frame_id_1:
        frame_id_2:
        flipval:
        max_clip_radius: take (max_clip_radius - 1) frames before
            and (max_clip_radius - 1) after the current frame from teh same clip.

    Returns:

    """
    ids = [frame_id_1, frame_id_2]
    clips = list()
    clip_radius = max_clip_radius
    for frame_id in ids:
        cur_clip_radius = 0
        while frame_id - cur_clip_radius >= 0 and frame_id + cur_clip_radius < len(seq_names) and \
                                seq_names[frame_id - cur_clip_radius] == seq_names[frame_id] == seq_names[frame_id + cur_clip_radius]:
            cur_clip_radius += 1
        clip_radius = min(clip_radius, cur_clip_radius)

    # if clip_len < max_clip_radius:
    #     print '{}, {}: clip_len={}'.format(frame_id_1, frame_id_2, clip_len)

    for frame_id in ids:
        clips.append(range(frame_id - clip_radius + 1, frame_id + clip_radius))
    assert len(clips[0]) == len(clips[1]) == 1 + (clip_radius - 1) * 2
    sim_values = list()
    for i, j in zip(*clips):
        sim_values.append(stacked_sim_matrix[i, j, int(flipval)])
    return np.mean(sim_values)


def clip_refine_sim(sim, seq_names, d, max_clip_radius=4):
    """
    Change in original matrix only the values which correspond to labeled pairs (for speed reasons).
    Args:
        sim:
        seq_names:
        d:
        max_clip_radius:

    Returns:

    """
    stacked_sim_matrix = np.stack([sim['simMatrix'], sim['simMatrix_flip']], axis=2)
    result_matrix = np.zeros_like(stacked_sim_matrix, dtype=np.float32)

    for anchor_idx in tqdm(xrange(len(d['anchors']))):
        anchor_id = d['anchors'][anchor_idx]
        for frame_id, flipval in zip(d['ids'][anchor_idx], d['flipvals'][anchor_idx]):
                result_matrix[anchor_id, frame_id, flipval] = \
                    compute_similarity_middle(stacked_sim_matrix, seq_names, anchor_id, frame_id, flipval, max_clip_radius=max_clip_radius)
    return dict(simMatrix=result_matrix[:, :, 0], simMatrix_flip=result_matrix[:, :, 1])


def compute_roc(d, sim, seq_names, max_clip_radius):
    stacked_sim_matrix = np.stack([sim['simMatrix'], sim['simMatrix_flip']], axis=2)
    assert stacked_sim_matrix.ndim == 3 and stacked_sim_matrix.shape[2] == 2
    assert stacked_sim_matrix[0, 0, 0] > stacked_sim_matrix[0, 0, 1]

    roc_auc_list = list()
    false_pos_rate_list = list()
    true_pos_rate_list = list()
    for i, anchor_id in enumerate(d['anchors']):
        scores = [compute_similarity_middle(stacked_sim_matrix, seq_names, anchor_id, frame_id, flipval, max_clip_radius=max_clip_radius)
                  for frame_id, flipval in zip(d['ids'][i], d['flipvals'][i].astype(int))]
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
        suffix = ''
        path_sim_matrix = join(dataset_root, 'sim/tf/', suffix, category,
                               'simMatrix_{}_tf_0.1conv_1fc_{}iter_{}_fc7_zscores.mat'.format(
                                   category, suffix + '_' if len(suffix) else '', iter_id))
    if not is_quiet:
        print 'Sim matrix path:', path_sim_matrix
    try:
        sim = scipy.io.loadmat(path_sim_matrix)
    except NotImplementedError:
        # matlab v7.3 file
        sim = dio.load(path_sim_matrix)

    seq_names = read_seq_names('OlympicSports', category)
    labels_path = join(dataset_root,
                       'dataset_labeling/labels_hdf5_19.02.16/labels_{}.hdf5'.format(
                           category))
    with h5py.File(labels_path, mode='r') as f:
        d = covert_labels_to_dict(f)

    max_clip_radius = 4
    avg_roc_auc, interp_roc_auc, roc_auc_list = compute_roc(d, sim, seq_names, max_clip_radius)
    print '{} n_acnhors: {} mean_ROC_AUC: {:.3f} interp_ROC_AUC: {:.3f}'.format(category, len(
        d['anchors']), avg_roc_auc, interp_roc_auc)

    sim = clip_refine_sim(sim, seq_names, d, max_clip_radius=max_clip_radius)
    hdf5storage.savemat('/export/home/asanakoy/workspace01/datasets/OlympicSports/sim/refined/'
                     'simMatrix_{0}_{0}_LR_0.001_M_0.9_BS_128_iter_20000_fc7_prerelu_refined_clipradius{1}_mean.mat'.format(category,
                                                                                                                           max_clip_radius),
                     sim, truncate_existing=True)

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
    aucs = list()
    for cat in categories:
        try:
            # path_sim_matrix = '/export/home/asanakoy/workspace/OlympicSports/sim/tf/{0}/simMatrix_{0}_tf_0.1conv_1fc_iter_20000_fc7_zscores.mat'.format(cat)
            path_sim_matrix = '/export/home/asanakoy/workspace/OlympicSports/sim/sim_matrices_miguel/CliqueCNN/simMatrix_{0}_{0}_LR_0.001_M_0.9_BS_128_iter_20000_fc7_prerelu.mat'.format(cat)
            roc_auc = compute_roc_auc_from_sim([cat], path_sim_matrix=path_sim_matrix, is_quiet=True)
            aucs.append(roc_auc)
        except IOError as e:
            print e
            print cat
    print 'Mean ROC AUC: \t{}'.format(np.mean(aucs))


if __name__ == '__main__':
    run_all_cat()
