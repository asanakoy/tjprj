# Test STLNet
# Copyright (c) 2016 Artsiom Sanakoyeu
import numpy as np
import time
import os
from scipy.misc import imread, imresize
import tensorflow as tf
import tfext.convnet
from trainhelper import trainhelper

MODELS_DIR = '/export/home/asanakoy/workspace/tfprj/data'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_feed_forward(net, im_shape):
    print 'test_feed_forward'
    batch = np.zeros((5,) + im_shape, dtype=np.float)
    for i in xrange(5):
        im = imread(os.path.join(DATA_DIR, "{}.png".format(i + 1)))[:, :, :3]
        im = imresize(im, im_shape[:2]).astype(np.float32)
        im = im - np.mean(im)
        batch[i, ...] = im
    batch = batch[:, :, :, ::-1]  # MAKE BGR!

    t = time.time()
    output = net.sess.run(net.prob, feed_dict={'input/x:0': batch, 'input/is_phase_train:0': False})
    for input_im_ind in range(output.shape[0]):
        print 'Image {}:  {}'.format(input_im_ind, output[input_im_ind])
    print time.time() - t


if __name__ == '__main__':
    params = {
        'device_id': '/gpu:0',
        'im_shape': (227, 227, 3),
        'num_classes': 100,
        'gpu_memory_fraction': 0.3,
        'random_init_type': tfext.convnet.Convnet.RandomInitType.XAVIER_GAUSSIAN
    }

    net = tfext.convnet.Convnet(**params)

    with net.graph.as_default():
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(net.fc6, net.y_gt))
        update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
        assert len(update_ops) > 0
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdagradOptimizer(learning_rate=1e-4,
                                                   initial_accumulator_value=0.00001).minimize(cross_entropy, global_step=net.global_iter_counter)
            # train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy,
            #                                                                      global_step=net.global_iter_counter)
        correct_prediction = tf.equal(tf.cast(tf.argmax(net.prob, 1), tf.int32), net.y_gt)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    net.sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    net.restore_from_alexnet_snapshot(trainhelper.get_alexnet_snapshot_path(), 5)


    test_feed_forward(net, params['im_shape'])

    batch = np.zeros((5,) + params['im_shape'], dtype=np.float)
    for i in xrange(5):
        im = imread(os.path.join(DATA_DIR, "{}.png".format(i + 1)))[:, :, :3]
        im = imresize(im, params['im_shape'][:2]).astype(np.float32)
        im = im - np.mean(im)
        batch[i, ...] = im
    batch = batch[:, :, :, ::-1]

    for i in range(750):
        y = [0, 1, 5, 9, 3]

        probs, train_accuracy, _, global_iter = net.sess.run([net.prob, accuracy, train_step, net.global_iter_counter],
                                                              feed_dict={net.x: batch,
                                                                         net.y_gt: y,
                                                                         net.dropout_keep_prob: 0.5,
                                                                         net.is_phase_train: True})
        test_probs, test_accuracy, global_iter = net.sess.run([net.prob, accuracy, net.global_iter_counter],
                                                   feed_dict={net.x: batch,
                                                   net.y_gt: y,
                                                   net.dropout_keep_prob: 1.0,
                                                   net.is_phase_train: False})
        np.set_printoptions(precision=2)
        print "step {}, training accuracy {:.2f}: {}".format(global_iter, train_accuracy, probs[:, 0])
        print "step {}, test accuracy     {:.2f}: {}".format(global_iter, test_accuracy, test_probs[:, 0])
        if i % 600 == 0:
            net.reset_fc6()

    saver = tf.train.Saver()
    checkpoint_prefix = os.path.expanduser('~/tmp/tmp-checkpoint-tensorflow-test')
    saver.save(net.sess, checkpoint_prefix, global_step=1, write_meta_graph=False)
    net.restore_from_snapshot(checkpoint_prefix + '-1', 6)
    net.reset_fc6()
