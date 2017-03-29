from unittest import TestCase
import os
import tensorflow as tf
import tfext.alexnet

_MODELS_DIR = '/export/home/asanakoy/workspace/tfprj/data'


class TestAlexnet(TestCase):
    def setUp(self):
        params = {
            'init_model': os.path.join(_MODELS_DIR, 'bvlc_alexnet.npy'),
            'num_classes': 1000,
            'device_id': '/gpu:0',
            'num_layers_to_init': 4,
            'im_shape': (227, 227, 3)
        }
        self.graph = tf.Graph()
        with self.graph.as_default():
            net = tfext.alexnet.Alexnet(**params)
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.fc8, labels=net.y_gt))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.cast(tf.argmax(net.prob, 1), tf.int32), net.y_gt)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.net = net

    def test_restore_from_other_snapshot(self):
        snapshot_path = '/export/home/asanakoy/workspace01/datasets/OlympicSports/cnn/cnn_tf_old/long_jump/checkpoint-20000'
        with self.graph.as_default():
            self.net.sess.run(tf.global_variables_initializer())
            with self.assertRaises(Exception):
                self.net.restore_from_snapshot(snapshot_path, 7)

    def test_restore_from_snapshot(self):
        snapshot_path = '/export/home/asanakoy/workspace/lsp/cnn/checkpoint-30000'
        with self.graph.as_default():
            self.net.sess.run(tf.global_variables_initializer())
            self.net.restore_from_snapshot(snapshot_path, 7)
