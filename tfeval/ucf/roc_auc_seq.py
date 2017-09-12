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

from tfeval.olympicsports.roc.roc_auc import get_roc_auc, covert_labels_to_dict, compute_interpolated_roc_auc


def read_seq_names(dataset, category):
    img_pathes_file = '/export/home/asanakoy/datasets/ucf_sports/data/image_data/imagePaths_{}.txt'.format(category)
    with open(img_pathes_file) as f:
        relative_img_pathes = f.readlines()
    seqnames = [path[:path.index('/')] for path in relative_img_pathes]
    return seqnames


def compute_similarity_before(stacked_sim_matrix, seq_names, frame_id_1, frame_id_2, flipval, MAX_CLIP_LEN):
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
    return np.min(sim_values)


def compute_similarity_middle(stacked_sim_matrix, seq_names, frame_id_1, frame_id_2, flipval,
                              max_clip_radius=4):
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
        sim_values.append(stacked_sim_matrix[i, j, flipval])
    return np.mean(sim_values)


def clip_refine_sim(sim, seq_names, d, max_clip_radius=4):
    stacked_sim_matrix = np.stack([sim['simMatrix'], sim['simMatrix_flip']], axis=2)
    result_matrix = np.zeros_like(stacked_sim_matrix, dtype=np.float32)

    for anchor_idx in tqdm(xrange(len(d['anchors']))):
        anchor_id = d['anchors'][anchor_idx]
        for frame_id, flipval in zip(d['ids'][anchor_idx], d['flipvals'][anchor_idx]):
                result_matrix[anchor_id, frame_id, int(flipval)] = \
                    compute_similarity_before(stacked_sim_matrix, seq_names, anchor_id, frame_id, int(flipval), max_clip_radius)
    return dict(simMatrix=result_matrix[:, :, 0], simMatrix_flip=result_matrix[:, :, 1])


def compute_roc(d, sim, seq_names, max_clip_radius):
    stacked_sim_matrix = np.stack([sim['simMatrix'], sim['simMatrix_flip']], axis=2)
    assert stacked_sim_matrix.ndim == 3 and stacked_sim_matrix.shape[2] == 2
    # assert stacked_sim_matrix[0, 0, 0] > stacked_sim_matrix[0, 0, 1]

    roc_auc_list = list()
    false_pos_rate_list = list()
    true_pos_rate_list = list()
    for i, anchor_id in enumerate(d['anchors']):
        scores = [compute_similarity_middle(stacked_sim_matrix, seq_names, anchor_id, frame_id, int(flipval), max_clip_radius)
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
    assert len(argv) == 2
    category = argv[0]
    ref_category = argv[1]

    dataset_root = '/export/home/asanakoy/datasets/ucf_sports'

    if not is_quiet:
        print 'Sim matrix path:', path_sim_matrix
    try:
        sim = scipy.io.loadmat(path_sim_matrix)
    except NotImplementedError:
        # matlab v7.3 file
        sim = dio.load(path_sim_matrix)

    if 'simMatrix_flip' not in sim:
        assert 'simMatrix_flipped' in sim
        sim['simMatrix_flip'] = sim['simMatrix_flipped']
        del sim['simMatrix_flipped']

    seq_names = read_seq_names('OlympicSports', category)
    labels_path = join(dataset_root,
                       'dataset_labeling/hdf5/labels_hdf5_11.01.17/labels_{}.hdf5'.format(
                           category))
    with h5py.File(labels_path, mode='r') as f:
        d = covert_labels_to_dict(f)

    max_clip_radius = 1
    avg_roc_auc, interp_roc_auc, roc_auc_list = compute_roc(d, sim, seq_names, max_clip_radius)
    print '{} n_acnhors: {} mean_ROC_AUC: {:.3f} interp_ROC_AUC: {:.3f}'.format(category, len(
        d['anchors']), avg_roc_auc, interp_roc_auc)

    # sim = clip_refine_sim(sim, seq_names, d, max_clip_radius=max_clip_radius)
    # hdf5storage.savemat('/export/home/asanakoy/datasets/ucf_sports/ours/sim_matrices_time_context/'
    #                  'simMatrix_{}_{}_LR_0.001_M_0.9_BS_128_iter_20000_fc7_prerelu_refined_clipradius{}_mean.mat'.format(category, ref_category,
    #                                                                                                                        max_clip_radius),
    #                  sim, truncate_existing=True)

    return interp_roc_auc


def run_all_cat():
    categories = [
        'Kicking',
        'Swing-Bench',
        'Swing-SideAngle',
        'Run-Side']
    ref_categories = [
        'hammer_throw',
        'hammer_throw',
        'diving_springboard_3m',
        'long_jump']
    aucs = list()
    for cat, ref_cat in zip(categories, ref_categories):
        try:
            path_sim_matrix = '/export/home/asanakoy/datasets/ucf_sports/ours/sim_matrices/simMatrix_{}_{}_LR_0.001_M_0.9_BS_128_iter_20000_fc7_prerelu.mat'.format(cat, ref_cat)
            # path_sim_matrix = '/export/home/asanakoy/datasets/ucf_sports/ours/sim_matrices_time_context/simMatrix_{}_{}_LR_0.001_M_0.9_BS_128_iter_20000_fc7_prerelu_refined_clipradius1_mean.mat'.format(cat, ref_cat)
            roc_auc = compute_roc_auc_from_sim([cat, ref_cat], path_sim_matrix=path_sim_matrix, is_quiet=True)
            aucs.append(roc_auc)
        except IOError as e:
            print e
            print cat
    print 'Mean ROC AUC: \t{}'.format(np.mean(aucs))


if __name__ == '__main__':
    run_all_cat()
