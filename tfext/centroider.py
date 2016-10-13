# --
# Copyright (c) 2016 Artsiom Sanakoyeu
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ++

import numpy as np
import scipy.spatial.distance as spdis
import sklearn.preprocessing as preprocess
import tensorflow as tf



class Centroider(object):
    """
    Calculate the closest centroid for each sample.
    """

    def __init__(self, net, batch_ldr):
        """

        :param batch_loader: BatchLoader
        :type batch_loader: BatchLoader
        :return:
        """
        print "Calculating nearest representatives for each class..."
        self.net = net
        self.batch_ldr = batch_ldr
        self.mu = np.zeros((np.max(batch_ldr.labels)+1, 4096))

    def updateCentroids(self, batch_size=128):
        """

        :param batch_size:
        :return:
        """

        for label in xrange(np.max(self.batch_ldr.labels) + 1):

            print "Represenative for class {}...".format(label)
            idx_class = np.nonzero(self.batch_ldr.labels == label)[0]
            image_ids = np.asarray(self.batch_ldr.indexlist)[idx_class]
            flip_values = np.asarray(self.batch_ldr.flip_values)[idx_class]
            unq_vals, unq_idx = np.asarray(np.unique(image_ids, return_index=True))

            features = np.zeros((unq_idx.shape[0], 4096))
            image = np.empty((unq_idx.shape[0], 3, 227, 227), dtype=np.float32)
            for idx, i in enumerate(unq_idx):
                image[idx, :, :, :] = self.batch_ldr.images_container.get_image(image_ids[i], flip_values[i])
            features = self.get_feature(image)
            self.mu[label] = self.get_centroid(features)


    def get_nearest_mu(self, labels):
        """
        Get nearest centroid
        """
        return self.mu[labels.astype('int32')]

    def get_feature(self, image):
        """
        :param image:
        :return: feature representation of that image

        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        image = image.transpose((0, 2, 3, 1))
        fc7 = self.net.sess.run(self.net.fc7, feed_dict={self.net.x: image})
        return tf.nn.l2_normalize(fc7, dim=1).eval(session=self.net.sess)



    def get_centroid(self, X):

        S = spdis.squareform(spdis.pdist(X, 'euclidean'))
        rowsum_S = np.sum(S, axis=0)
        return X[rowsum_S.argmin()]






