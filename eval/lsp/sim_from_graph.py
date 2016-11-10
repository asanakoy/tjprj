import tensorflow as tf
from collections import namedtuple
import os
from os.path import join
import numpy as np
from eval.image_getter import ImageGetterFromPaths
import eval.olympicsports.roc.roc_auc
import eval.features
import eval.olympicsports.utils
import glob

Net = namedtuple('Net', ['sess', 'fc7', 'fc6', 'maxpool5', 'conv5', 'graph'])


def get_sim_output_path(model_name, iteration, params):
    p = join('/export/home/asanakoy/workspace/lsp/sim/tf',
             params['category'], 'simMatrix_{}_{}{}_iter_{}_{}_{}.mat'.
             format(params['category'], model_name,
                    '_nomean' if params['mean'] is None else '',
                    iteration, ''.join(params['layer_names']), params['norm_method']))
    return p


def load_net(snapshot_path, gpu_memory_fraction=None,
             conv5=None,
             maxpool5=None,
             fc6=None,
             fc7=None):
    graph_path = snapshot_path + '.meta'
    if not os.path.exists(graph_path):
        raise IOError('Graph meta file not found: {}'.format(graph_path))
    if not os.path.exists(snapshot_path):
        raise IOError('Snapshot file not found: {}'.format(snapshot_path))

    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    if gpu_memory_fraction is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=config)
        new_saver = tf.train.import_meta_graph(graph_path)
        new_saver.restore(sess, snapshot_path)

        return Net(sess=sess, graph=graph,
                   conv5=None if conv5 is None else tf.get_default_graph().get_tensor_by_name(conv5),
                   maxpool5=None if maxpool5 is None else tf.get_default_graph().get_tensor_by_name(maxpool5),
                   fc6=None if fc6 is None else tf.get_default_graph().get_tensor_by_name(fc6),
                   fc7=None if fc7 is None else tf.get_default_graph().get_tensor_by_name(fc7))


if __name__ == '__main__':

    for iteration in [60001]:
        crops_dir = '/export/home/asanakoy/workspace/lsp/crops_227x227'
        image_paths = [join(crops_dir, p) for p in glob.glob1(crops_dir, '*.png')]
        im_shape = (227, 227)
        model_name = 'joint_categories_0.1conv_anchors'
        # iteration = 80002
        snapshot_path = join('/export/home/asanakoy/workspace/OlympicSports/cnn/',
                               model_name, '', 'checkpoint-{}'.format(iteration))

        category = ''
        mean_path = join(
            '/export/home/asanakoy/workspace/OlympicSports/cnn/', model_name, category, 'mean.npy')
        mean = np.load(mean_path)

        params = {
            'category': '',
            'layer_names': ['conv5'],
            'norm_method': None,
            'image_getter': ImageGetterFromPaths(image_paths, im_shape),
            'mean': None,
            'im_shape': (227, 227),
            'batch_size': 256,
            'use_batch_norm': False,
            'gpu_memory_fraction': 0.35,
            'device_id': '/gpu:{}'.format(0)
        }

        sim_output_path = get_sim_output_path(model_name, iteration, params)

        net = load_net(snapshot_path, gpu_memory_fraction=0.4,
                       conv5='conv5/conv:0',
                       maxpool5='conv5/maxpool:0',
                       fc6='fc6/fc:0',
                       fc7='fc7/fc:0'
                       )

        print 'Using Snapshot:', snapshot_path
        print 'Output sim matrix to', sim_output_path
        eval.features.compute_sim_and_save(sim_output_path, net=net, **params)
        net.sess.close()
        print params['layer_names']
