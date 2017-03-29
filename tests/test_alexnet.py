# Copyright (c) 2016 Artsiom Sanakoyeu
import numpy as np
import time
import os
from alexnet_classes import class_names
from scipy.misc import imread
import tensorflow as tf
import tfext.alexnet

MODELS_DIR = '/export/home/asanakoy/workspace/tfprj/data'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_feed_forward(net):
    print 'test_feed_forward'
    batch = np.zeros((5, 227, 227, 3), dtype=np.float)
    for i in xrange(5):
        im = (imread(os.path.join(DATA_DIR, "{}.png".format(i + 1)))[:, :, :3]).astype(
            np.float32)
        im = im - np.mean(im)
        batch[i, ...] = im
    batch = batch[:, :, :, ::-1]  # MAKE BGR!

    t = time.time()
    output = net.sess.run(net.prob, feed_dict={'input/x:0': batch, 'input/is_phase_train:0': False})
    for input_im_ind in range(output.shape[0]):
        inds = np.argsort(output)[input_im_ind, :]
        print "Image", input_im_ind
        print 'class {}'.format(inds[-1])
        for i in range(5):
            print class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]
    print time.time() - t


if __name__ == '__main__':
    params = {
        'init_model': os.path.join(MODELS_DIR, 'bvlc_alexnet.npy'),
        'num_classes': 1000,
        'device_id': '/gpu:0',
        'num_layers_to_init': 6,
        'im_shape': (227, 227, 3),
        'use_batch_norm': False,
        'gpu_memory_fraction': 0.5
    }

    net = tfext.alexnet.Alexnet(**params)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.fc8, labels=net.y_gt))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=net.global_iter_counter)
    # train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.cast(tf.argmax(net.prob, 1), tf.int32), net.y_gt)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    net.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(net.sess, '/export/home/asanakoy/tmp/alexnet-tf-test')
    saver.restore(net.sess, '/export/home/asanakoy/tmp/alexnet-tf-test')

    test_feed_forward(net)

    batch = np.zeros((5, 227, 227, 3), dtype=np.float)
    for i in xrange(5):
        im = (imread(os.path.join(DATA_DIR, "{}.png".format(i + 1)))[:, :, :3]).astype(np.float32)
        im = im - np.mean(im)
        batch[i, ...] = im
    batch = batch[:, :, :, ::-1]

    for i in range(100):
        # batch = np.random.random((10, 227, 227, 3))
        y = [205, 344, 356, 1, 84]
        # y = label_binarize(y, classes=range(1000))

        if i % 1 == 0:
            _, global_iter = net.sess.run([train_step, net.global_iter_counter],
                                          feed_dict={net.x: batch,
                                                     net.y_gt: y,
                                                     net.fc6_keep_prob: 0.5,
                                                     net.fc7_keep_prob: 0.5,
                                                     net.is_phase_train: True})

            train_accuracy, global_iter = net.sess.run([accuracy, net.global_iter_counter],
                                                       feed_dict={net.x: batch,
                                                       net.y_gt: y,
                                                       net.fc6_keep_prob: 1.0,
                                                       net.fc7_keep_prob: 1.0,
                                                       net.is_phase_train: False})
            print("step %d, training accuracy %f" % (global_iter, train_accuracy))

    # from trainhelper.trainhelper import get_sim
    # d = get_sim(net, 'long_jump', ['fc7'], return_features=False)
    # print d.keys()

    # print("test accuracy %g" % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    pass