# Copyright (c) 2016 Artsiom Sanakoyeu
import numpy as np
import time
import os
from scipy.misc import imread
import tensorflow as tf
import tfext.alexnet
from trainhelper import trainhelper

MODELS_DIR = '/export/home/asanakoy/workspace/tfprj/data'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DATA_PATH = os.path.join(MODELS_DIR, 'bvlc_alexnet.npy')
SNAPSHOT_PREFIX_PATH = '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.tf'


def test_snaphot():
    params = {
        'init_model': None,
        'num_classes': 1000,
        'device_id': '/gpu:0',
        'num_layers_to_init': 0,
        'im_shape': (227, 227, 3),
        'use_batch_norm': False,
        'gpu_memory_fraction': 0.27
    }
    with tf.Graph().as_default():
        net = tfext.alexnet.Alexnet(**params)
        net.sess.run(tf.global_variables_initializer())
        checkpoint_prefix = SNAPSHOT_PREFIX_PATH
        saver = tf.train.Saver()
        saver.restore(net.sess, checkpoint_prefix)

        net_data = np.load(DATA_PATH).item()

        for i in xrange(1, 6):
            name = 'conv{}'.format(i)
            print name
            w = net.graph.get_tensor_by_name('conv{}/weight:0'.format(i))
            b = net.graph.get_tensor_by_name('conv{}/bias:0'.format(i))

            w, b = net.sess.run([w, b])
            assert np.all(w == net_data[name]['weights'])
            assert np.all(b == net_data[name]['biases'])
        for i in xrange(6, 9):
            name = 'fc{}'.format(i)
            print name
            w = net.graph.get_tensor_by_name('fc{}/weight:0'.format(i))
            b = net.graph.get_tensor_by_name('fc{}/bias:0'.format(i))

            w, b = net.sess.run([w, b])
            assert np.all(w == net_data[name]['weights'])
            assert np.all(b == net_data[name]['biases'])
        net.sess.close()


def convert():
    params = {
        'init_model': DATA_PATH,
        'num_classes': 1000,
        'device_id': '/gpu:0',
        'num_layers_to_init': 8,
        'im_shape': (227, 227, 3),
        'use_batch_norm': False,
        'gpu_memory_fraction': 0.27
    }
    with tf.Graph().as_default():
        net = tfext.alexnet.Alexnet(**params)
        net.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkpoint_prefix = SNAPSHOT_PREFIX_PATH
        saver.save(net.sess, checkpoint_prefix, write_meta_graph=True)
        print 'Saved'
        net.sess.close()


if __name__ == '__main__':
    convert()
    test_snaphot()
