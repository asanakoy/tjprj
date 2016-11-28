import numpy as np
import tensorflow as tf
import scipy.spatial.distance as spdis
import scipy.stats as stats
import sklearn.metrics.classification as metrics
from stl10_input import read_all_images
from stl10_input import read_labels
from pprint import pformat
import os
import tfext
import scipy.io as sio
import h5py
import PIL.Image as Image

import tfext.alexnet
import tfext.caffenet
import tfext.utils
import scipy.io as sio

class supervised_evaluation(object):

    def __init__(self, net):
        """
        Read data and initilize fields
        :param net:
        :return:
        """

        # Paths to data
        path_test_labels = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/classification/test_label_and_perm.mat'
        path_test_images = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/images_CVPR17/images_test.mat'
        path_train_labels = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/classification/training_data_labels.mat'
        path_train_images = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/images_CVPR17/images.mat'

        # Load training/test data
        train_samples = h5py.File(path_train_images, 'r')
        train_labels = sio.loadmat(path_train_labels)
        test_samples = h5py.File(path_test_images, 'r')
        test_labels = sio.loadmat(path_test_labels)

        # Parse as ndarray
        train_samples = train_samples['images'][()]
        train_samples = np.transpose(train_samples, [0, 2, 3, 1])
        test_samples = test_samples['images'][()]
        test_samples = np.transpose(test_samples, [0, 2, 3, 1])
        train_labels = train_labels['labels'][0]
        test_labels_aux = test_labels['test_label'][0]

        self.train_samples = train_samples
        self.train_labels = train_labels-1
        self.test_samples = test_samples
        self.test_labels = test_labels_aux-1

        # Set net through function
        self.net = None
        self.update_net(net)

    def train(self, n_iters_training, train_op, loss, output_dir, saver, summary, summary_writter, warmup_iters, conv_lr):
        """
        Run fine tuning of the current network (freezing all layers but the last fc)
        :param params:
        :return:
        """

        acc = []
        with tf.Graph().as_default():

            img_augmenter = self.create_augmenter()
            for step in range(n_iters_training):

                samples_batch, labels_batch = self.getBatch(batch_size=128)
                samples_batch = img_augmenter.augment_batch(samples_batch)
                keep_prob = 0.5
                is_phase_train = True
                feed_dict = {
                    self.net.x: samples_batch,
                    self.net.y_gt: labels_batch,
                    self.net.dropout_keep_prob: keep_prob,
                    'input/is_phase_train:0': is_phase_train
                }

                if step >= warmup_iters:
                    feed_dict['lr/conv_lr:0'] = conv_lr
                else:
                    feed_dict['lr/conv_lr:0'] = 0.0


                _, summary_str, loss_value = self.net.sess.run([train_op, summary, loss], feed_dict=feed_dict)
                print "Supervised eval train step {}: loss = {} ".format(step, loss_value)

                if step % 200 == 0:
                    summary_writter.add_summary(summary_str, global_step=step)
                    # summary_writer.flush()

                if (step % 10000) == 0:
                    acc_aux = self.test()
                    acc.append(acc_aux)
                    print acc

                    # Save the current net to snapshot
                    checkpoint_file = os.path.join(output_dir, 'checkpoint_before_evaluation')
                    saver.save(self.net.sess, checkpoint_file, global_step=step)
                    summary_writter.add_summary(tfext.utils.create_sumamry('Supervised Acc', acc_aux),
                                                         global_step=step)
                    summary_writter.flush()

    def test(self):
        """
        Evaluate network on the the test set  and return acc
        :param params:
        :return:
        """
        pred_labels = []
        for idx, test_sample in enumerate(self.test_samples):
            print 'Testing sample {}/{}'.format(idx, self.test_samples.shape[0])
            keep_prob = 1
            is_phase_train = False
            feed_dict = {
                self.net.x: np.expand_dims(test_sample, 0),
                self.net.dropout_keep_prob: keep_prob,
                'input/is_phase_train:0': is_phase_train
            }
            pred = self.net.sess.run(self.net.prob_stl10, feed_dict=feed_dict).argmax()
            pred_labels.append(pred)

        acc = np.float32(sum(self.test_labels == np.asarray(pred_labels))) / self.test_samples.shape[0]
        print "Accuracy on evaluation set {}".format(acc)
        return acc

    def getBatch(self, batch_size=128):
        """
        Get batch to train from supervised data
        :param batch_size:
        :return:
        """
        batch_idx = np.random.choice(self.train_labels.shape[0], batch_size, replace=False)
        samples_batch = self.train_samples[batch_idx, :, :, :]
        labels_batch = self.train_labels[batch_idx]
        return samples_batch, labels_batch


    def update_net(self, net):
        """
        Wrapper for net updating for security
        :param net:
        :return:
        """
        # assert net is not None, "Updated net to none"
        self.net = net

    def create_augmenter(self):
        default_params = dict(hflip=True, vflip=False,
                              scale_to_percent=(1, 1),
                              scale_axis_equally=True,
                              rotation_deg=5, shear_deg=5,
                              translation_x_px=10, translation_y_px=10,
                              interpolation_order=1,
                              channel_is_first_axis=False,
                              preserve_range=True)
        params = default_params
        print 'Creating ImageAugmenter with params:\n{}'.format(pformat(params))
        import imageaugmenter.ImageAugmenter
        augmenter = \
            imageaugmenter.ImageAugmenter(96, 96, **params)
        augmenter.pregenerate_matrices(1000)
        return augmenter

    def compute_sim_matrix(self):
        """
        Compute sim Matrix using the current
        :return:
        """
        print "Computing similarity matrix for NN evaluation..."
        ndims = np.prod(np.asarray(self.net.fc6._shape_as_list()[1:]))

        features_test = np.zeros((self.test_samples.shape[0], ndims))
        for idx, test_sample in enumerate(self.test_samples):
            test_sample = Image.fromarray(test_sample)
            test_sample = test_sample.resize((227, 227), Image.ANTIALIAS)
            test_sample = np.asarray(test_sample)
            keep_prob = 1
            is_phase_train = False
            feed_dict = {
                self.net.x: np.expand_dims(test_sample, 0),
                self.net.fc6_keep_prob: keep_prob,
                self.net.fc7_keep_prob: keep_prob,
                'input/is_phase_train:0': is_phase_train
            }
            features_test[idx, :] = self.net.sess.run(self.net.fc6, feed_dict=feed_dict).flatten()

        features_train = np.zeros((self.train_samples.shape[0], ndims))
        for idx, train_sample in enumerate(self.train_samples):
            train_sample = Image.fromarray(train_sample)
            train_sample = train_sample.resize((227, 227), Image.ANTIALIAS)
            train_sample = np.asarray(train_sample)
            keep_prob = 1
            is_phase_train = False
            feed_dict = {
                self.net.x: np.expand_dims(train_sample, 0),
                self.net.fc6_keep_prob: keep_prob,
                self.net.fc7_keep_prob: keep_prob,
                'input/is_phase_train:0': is_phase_train
            }
            features_train[idx, :] = self.net.sess.run(self.net.fc6, feed_dict=feed_dict).flatten()

        simMatrix_aux = spdis.cdist(features_test, features_train,
                                    'correlation')
        self.sim_matrix = np.float32(2.0 - simMatrix_aux)

        # pathtosim = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/VOC/classification/sims/simMatrix_VOC_iter_1_bvlc_alexnet_fc6_0.mat'
        # s = sio.loadmat(pathtosim)
        # self.sim_matrix = s['sim_matrix']

    def testing_sample_selection(self):
        """
        This function selects k at max 50 random samples for testing from all classes
        :param n:
        :return:
        """

        return np.arange(self.test_labels.shape[0])

    def nn_classifier(self, n):
        """
        Calculate n nearest neighbours
        :param n: nearest neighbours
        :return:
        """

        # Get NNs indices for every test sample
        sorted_nns = np.zeros((self.sim_matrix.shape[0], n))
        sorted_nns_score = np.zeros((self.sim_matrix.shape[0], n))
        for idx_row, sim_row in enumerate(self.sim_matrix):
            perm = sim_row.argsort()
            idxs_nns = perm[::-1][:n]
            sorted_nns[idx_row, :] = idxs_nns
            sorted_nns_score[idx_row, :] = self.sim_matrix[idx_row, idxs_nns]

        # Initialize linear decaying weights for sampels and pred_labels)
        pred_labels = np.zeros((self.sim_matrix.shape[0], self.test_labels.max() + 1))

        # Loop over test samples disregarding self similarity and vote for each label with a weight depending
        # on the position of the NN
        for test_sample in xrange(self.sim_matrix.shape[0]):
            votes_per_label = dict.fromkeys(np.unique(self.test_labels), 0.0)
            scores_per_label = dict.fromkeys(np.unique(self.test_labels), 0.0)
            for idx, nn in enumerate(sorted_nns[test_sample, :]):
                votes_per_label[np.int32(self.train_labels[np.int32(nn)])] += 1
                scores_per_label[np.int32(self.train_labels[np.int32(nn)])] += sorted_nns_score[test_sample, idx]

            norm_votes = np.asarray(votes_per_label.values()) / np.asarray(votes_per_label.values()).sum()
            norm_scores = np.asarray(scores_per_label.values()) / np.asarray(scores_per_label.values()).sum()
            pred_labels[test_sample, :] = norm_votes * norm_scores

        # Get maximal predicted label for each test sample and return acc
        pred_labels = pred_labels.argmax(axis=1)
        acc = metrics.accuracy_score(y_pred=pred_labels, y_true=self.test_labels)
        return acc

    def ten_fold_cv_nn_eval(self, n):
        """
        Run experiment 10 times reporting mean and std
        :param n: number of neighbours to calculate prediction
        :return:
        """
        self.compute_sim_matrix()
        acc = []
        for i in range(1):
            print "Evaluating NN retrieval fold {}/{}".format(i, 10)
            acc.append(self.nn_classifier(n))
        return np.asarray(acc).mean()

if __name__ == "__main__":



    # path_to_snapshot = '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.npy'
    # net = tfext.alexnet.Alexnet(init_model=path_to_snapshot,
    #                             random_init_type=tfext.alexnet.Alexnet.RandomInitType.XAVIER_GAUSSIAN,
    #                             num_layers_to_init=7, num_classes=1000)

    path_to_snapshot = '/export/home/mbautist/Desktop/workspace/cnn_similarities/tjprj/data/pascal_bbox/model_ours_87k.npy'
    net = tfext.alexnet.Alexnet(init_model=path_to_snapshot,
                                  random_init_type=tfext.caffenet.CaffeNet.RandomInitType.XAVIER_GAUSSIAN,
                                  num_classes=1000, num_layers_to_init=6)


    net.sess.run(tf.initialize_all_variables())
    ev = supervised_evaluation(net=net)
    acc = ev.ten_fold_cv_nn_eval(10)
    print acc
