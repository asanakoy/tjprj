import os.path
import sys
from demo.train import get_num_classes, run_training


def main(argv):
    if len(argv) == 0:
        argv = ['0']
    dataset = 'lsp'

    images_aug_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/lsp_dataset_original/augmented_data/10T/training_data_lsp_dataset_original_LSP.mat'
    indices_dir = '/export/home/mbautist/Desktop/workspace/cnn_similarities/MIL-CliqueCNN/clustering/LSP/iter_1/'
    train_indices_path = os.path.join(indices_dir, 'train_indices.mat')
    mean_path = os.path.join(indices_dir, 'mean.npy')

    output_dir = os.path.expanduser('~/workspace/lsp/cnn')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_cliques = get_num_classes(train_indices_path)

    params = {
        'im_shape': (227, 227, 3),
        'batch_size': 128,
        'base_lr': 0.001,
        'fc_lr_mult': 1.0,
        'conv_lr_mult': 0.1,
        'num_layers_to_init': 6,
        'dataset': dataset,
        'num_classes': num_cliques,
        'snapshot_iter': 10000,
        'max_iter': 30000,
        'indexing_1_based': 1,
        'images_mat_filepath': images_aug_path,
        'indexfile_path': train_indices_path,
        'mean_filepath': mean_path,
        'seed': 1988,
        'output_dir': output_dir,
        'init_model': '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.npy',
        'device_id': '/gpu:{}'.format(int(argv[0]))
    }
    run_training(**params)


if __name__ == '__main__':
    main(sys.argv[1:])
