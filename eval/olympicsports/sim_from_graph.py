import tensorflow as tf
from collections import namedtuple
import os
from os.path import join
import numpy as np
from eval.image_getter import ImageGetterFromMat
import eval.olympicsports.roc.roc_auc
import eval.features
import eval.olympicsports.utils
from utils import get_sim_pathes

Net = namedtuple('Net', ['sess', 'fc6', 'maxpool5', 'graph'])


def load_net(snapshot_path, gpu_memory_fraction=None):
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

        fc6 = tf.get_default_graph().get_tensor_by_name('fc6/fc6:0')
        maxpool5 = tf.get_default_graph().get_tensor_by_name('maxpool5:0')
        return Net(sess=sess, fc6=fc6, maxpool5=maxpool5, graph=graph)


if __name__ == '__main__':
    category = 'long_jump'

    mat_path = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/' + category + '/images.mat'
    model_name = 'convnet_scratch'

    mean_path = join(
        '/export/home/asanakoy/workspace/OlympicSports/cnn/', model_name, category, 'mean.npy')
    mean = np.load(mean_path)

    params = {
        'category': category,
        'number_layers_restore': 6,
        'layer_names': ['maxpool5'],
        'norm_method': None,
        'image_getter': ImageGetterFromMat(mat_path),
        'mean': mean,
        'im_shape': (227, 227),
        'batch_size': 256,
        'use_batch_norm': False,
        'gpu_memory_fraction': 0.35,
        'device_id': '/gpu:{}'.format(0)
    }
    snapshot_path, sim_output_path = get_sim_pathes(model_name,
                                                              iteration=None,
                                                              round_id=1,
                                                              **params)
    net = load_net(snapshot_path, gpu_memory_fraction=0.3)
    print 'Using Snapshot:', snapshot_path
    print 'Output sim matrix to', sim_output_path
    eval.features.compute_sim_and_save(sim_output_path, net=net, **params)
    net.sess.close()
    print params['layer_names']
    eval.olympicsports.roc.roc_auc.compute_roc_auc_from_sim([category], path_sim_matrix=sim_output_path)
