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


class Centroider(object):
    """
    Calculate the closest centroid for each sample.
    """

    def __init__(self, batch_ldr):
        """

        :param batch_loader: BatchLoader
        :type batch_loader: BatchLoader
        :return:
        """
        print "Calculating nearest representatives for each class..."
        self.batch_ldr = batch_ldr
        self.mu = np.zeros((np.max(batch_ldr.labels)+1, 4096))
        self.sigma = np.zeros((np.max(batch_ldr.labels)+1))

    def updateCentroids(self, sess, in_ph, out_ph, batch_size=128):
        """

        :param batch_size:
        :return:
        """

        for label in np.unique(self.batch_ldr.labels):

            print "Represenative for class {}...".format(label)

            idx_class = np.nonzero(self.batch_ldr.labels == label)[0]
            image_ids = np.asarray(self.batch_ldr.indexlist)[idx_class]
            flip_values = np.asarray(self.batch_ldr.flip_values)[idx_class]
            unq_vals, unq_idx = np.asarray(np.unique(image_ids, return_index=True))
            features = np.zeros((unq_idx.shape[0], 4096))
            image = np.empty((unq_idx.shape[0], 3, 227, 227), dtype=np.float32)
            for idx, i in enumerate(unq_idx):
                image[idx, :, :, :] = self.batch_ldr.images_container.get_image(image_ids[i], flip_values[i])
            features = self.get_feature(image, sess, in_ph, out_ph)
            self.mu[label] = self.get_centroid(features)
            self.sigma[label] = self.get_deviation(features, label)


    def get_nearest_mu(self, labels):
        """
        Get nearest centroid
        """
        return self.mu[labels.astype('int32')]

    def get_sigma(self, labels):
        return np.expand_dims(np.mean(self.sigma[labels.astype('int32')]), 0)

    def get_feature(self, image, sess, in_ph, out_ph):
        """
        :param image:
        :return: feature representation of that image

        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        image = image.transpose((0, 2, 3, 1))
        features = sess.run(out_ph, feed_dict={in_ph: image})
        features = preprocess.normalize(features)
        return features

    def get_deviation(self, X, label):

        diff = X - self.mu[label]
        norm2 = np.sum(np.abs(diff) ** 2, axis=-1)
        return np.mean(norm2)

    def get_centroid(self, X):

        S = spdis.squareform(spdis.pdist(X, 'euclidean'))
        rowsum_S = np.sum(S, axis=0)
        return X[rowsum_S.argmin()]






