import sys
from os.path import join
import glob
import numpy as np

from demo.train import get_num_classes
from eval.image_getter import ImageGetterFromPaths
import eval.features


def main(argv):
    if len(argv) == 0:
        gpu_id = 0
    else:
        gpu_id = int(argv[0])

    crops_dir = '/export/home/asanakoy/workspace/lsp/crops_227x227'
    image_paths = [join(crops_dir, p) for p in glob.glob1(crops_dir, '*.png')]

    indices_dir = '/export/home/mbautist/Desktop/workspace/cnn_similarities/MIL-CliqueCNN/clustering/LSP/iter_1/'
    train_indices_path = join(indices_dir, 'train_indices.mat')
    mean_path = join(indices_dir, 'mean.npy')
    mean = np.load(mean_path)
    num_classes = get_num_classes(train_indices_path)

    iteration = 30000
    init_model = '/export/home/asanakoy/workspace/lsp/cnn/checkpoint-{}'.format(iteration)
    model_name = 'tf_0.1conv_1fc_{}it'.format(iteration)
    im_shape = (227, 227)

    params = {
        'model_name': model_name,
        'category': '',
        'iter': iteration,
        'layer_names': ['fc6'],
        'image_getter': ImageGetterFromPaths(image_paths, im_shape),
        'mean': None, #mean,
        'im_shape': im_shape,
        'batch_size': 256,
        'num_classes': num_classes,
        'snapshot_path': init_model,
        'sim_output_dir': join(
            '/export/home/asanakoy/workspace/lsp/sim/tf'),
        'gpu_memory_fraction': 0.35,
        'device_id': '/gpu:{}'.format(gpu_id)
    }
    eval.features.compute_sim_and_save(**params)


if __name__ == '__main__':
    main(sys.argv[1:])
