import sys
from os.path import join
import glob
import numpy as np
import tensorflow as tf

from eval.image_getter import ImageGetterFromPaths
import eval.features
from  trainhelper import trainhelper


def get_sim_output_path(model_name, iteration, params):
    p = join('/export/home/asanakoy/workspace/lsp/sim/tf',
                params['category'], 'simMatrix_{}_{}{}_iter_{}_{}_{}.mat'.
                format(params['category'], model_name, '_nomean' if params['mean'] is None else '',
                iteration, ''.join(params['layer_names']), params['norm_method']))
    return p



def main(argv):
    if len(argv) == 0:
        gpu_id = 0
    else:
        gpu_id = int(argv[0])

    crops_dir = '/export/home/asanakoy/workspace/lsp/crops_227x227'
    image_paths = [join(crops_dir, p) for p in glob.glob1(crops_dir, '*.png')]

    # mean_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/MIL-CliqueCNN/clustering/LSP/iter_1/mean.npy'
    # mean_path = '/export/home/asanakoy/workspace/OlympicSports/cnn/joint_categories/mean.npy'
    # mean = np.load(mean_path)

    iteration = 0
    # init_model = '/export/home/asanakoy/workspace/OlympicSports/cnn/joint_categories_0.1conv_anchors/checkpoint-{}'.format(iteration)
    # init_model = trainhelper.get_alexnet_snapshot_path()
    # init_model = '/export/home/asanakoy/workspace/OlympicSports/cnn_2/2_rounds_aug_bbox_sq/hammer_throw/checkpoint-{}'.format(iteration)
    # model_name = '2_rounds_aug_bbox_sq_hammer_throw'.format(iteration)
    model_name = 'caffenet'
    im_shape = (227, 227)

    params = {
        'model_name': model_name,
        'category': '',
        # 'number_layers_restore': 6,
        'layer_names': ['fc6'],
        'norm_method': None,
        'image_getter': ImageGetterFromPaths(image_paths, im_shape),
        'mean': None,#mean,
        'im_shape': im_shape,
        'batch_size': 256,
        'snapshot_path': None,
        'gpu_memory_fraction': 0.4,
        'device_id': '/gpu:{}'.format(gpu_id)
    }
    sim_output_path = get_sim_output_path(model_name, iteration, params)

    import tfext.caffenet
    net_params = {
        'init_model': '/export/home/asanakoy/workspace/tfprj/data/bvlc_reference_caffenet/weights.npy',
        'num_classes': 1000,
        'device_id': '/gpu:0',
        'num_layers_to_init': 8,
        'im_shape': (227, 227, 3),
        'use_batch_norm': False,
        'gpu_memory_fraction': 0.5
    }

    net = tfext.caffenet.CaffeNet(**net_params)
    net.sess.run(tf.global_variables_initializer())
    eval.features.compute_sim_and_save(sim_output_path, norm_method=params.pop('norm_method'), net=net, **params)


if __name__ == '__main__':
    main(sys.argv[1:])
