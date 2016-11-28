from mercurial.minirst import replace

import numpy as np
import sklearn.svm as svm
import sklearn.metrics as eval
import scipy.spatial.distance as spdis


class KshotClassifier(object):
    """
    This class is a helper to evaluate results on the Caltech datasets
    """

    def __init__(self, x, y):
        """
        Initialize with samples and labels
        :return:
        """

        self.samples = x
        self.labels = y

    def training_sample_selection(self, k):
        """
        This function selects k training samples at random and at max 50 random samples for testing from all classes
        :param n:
        :return:
        """

        idxs_train = np.empty(0)
        idxs_test = np.empty(0)
        for class_idx in xrange(self.labels.max()):
            idxs_class = np.where(self.labels == class_idx)[0]
            idxs_train = np.append(idxs_train, np.random.choice(idxs_class, k, replace=False)).astype(dtype=np.int32)
            idxs_test = np.setdiff1d(idxs_class, idxs_train)
            idxs_test = np.append(idxs_test,
                                  np.random.choice(idxs_test, np.min([idxs_test.shape[0], 50]), replace=False))

        return idxs_train, idxs_test

    def train_ova_classifier(self, k):
        """
        This function trains a one-vs-all classifier on k samples from each class
        :param k:
        :return:
        """
        assert k <= 30, 'k can only be < 30'

        # Select k train samples for each class and train and OVR classifier
        idxs_training, idxs_test = self.training_sample_selection(k)
        idxs_val = np.random.choice(idxs_test, 10, replace=False)

        c_range = np.linspace(1, 100000, 100)
        acc_val = np.zeros(c_range.shape[0])

        # Cross validate C parameter
        for idx, c in enumerate(c_range):
            h = svm.LinearSVC(C=c, loss='hinge', multi_class='ovr')
            h.fit(self.samples[idxs_training], self.labels[idxs_training])
            pred_val_labels = h.predict(self.samples[idxs_val])
            acc_val[idx] = eval.accuracy_score(y_pred=pred_val_labels, y_true=self.labels[idxs_val])

        # Predict and get accuracy on test set
        h = svm.LinearSVC(C=c_range[acc_val.argmax()], loss='hinge', multi_class='ovr')
        h.fit(self.samples[idxs_training], self.labels[idxs_training])
        pred_labels = h.predict(self.samples[idxs_test])
        acc = eval.accuracy_score(y_pred=pred_labels, y_true=self.labels[idxs_test])
        return acc

    def ten_fold_cv_train(self, k):
        """
        Run experiment 10 times
        :param k:
        :return:
        """
        acc = []
        for i in range(10):
            acc.append(self.train_ova_classifier(k))
        return np.asarray(acc).mean(), np.asarray(acc).std()


class ZeroShotClassifier(object):
    """
    This class is a helper to evaluate results on the Caltech datasets, 0 shot learning requires a sim_matrix for NN
    retrieval
    """

    def __init__(self, sim_matrix, y):
        """
        Initialize with similarity matrix and labels
        :return:
        """


        # if sim_matrix is not input, calculate it
        if sim_matrix.shape[0] != sim_matrix.shape[1]:
            print "Initialized with features, calculating sim matrix..."
            sim_matrix = np.float32(2.0 - spdis.squareform(spdis.pdist(sim_matrix, 'correlation')))
        self.sim_matrix = sim_matrix
        self.labels = y

    def testing_sample_selection(self):
        """
        This function selects k at max 50 random samples for testing from all classes
        :param n:
        :return:
        """

        # Loop over classes selecting at max 50 samples for retrieving NNs
        idxs_test = np.empty(0)
        for class_idx in xrange(self.labels.max()):
            idxs_class = np.where(self.labels == class_idx)[0]
            idxs_test = np.append(idxs_test,
                                  np.random.choice(idxs_class, np.min([idxs_class.shape[0], 25]), replace=False))

        return idxs_test.astype(dtype=np.int32)

    def nn_classifier(self, n):
        """
        Calculate n nearest neighbours
        :param n: nearest neighbours
        :return:
        """

        # Get NNs indices for every test sample
        idxs_test = self.testing_sample_selection()
        nns = self.sim_matrix[idxs_test, :]
        sorted_nns = np.zeros((nns.shape[0], n))
        for idx, nn in enumerate(nns):
            sorted_nns[idx, :] = nn.argsort()[::-1][:n]

        # Initialize linear decaying weights for sampels and pred_labels
        weight = np.linspace(1, 0, n)
        pred_labels = np.zeros((idxs_test.shape[0], self.labels.max() + 1))

        # Loop over test samples disregarding self similarity and vote for each label with a weight depending
        # on the position of the NN
        for test_sample in xrange(idxs_test.shape[0]):
            votes_per_label = dict.fromkeys(np.unique(self.labels), 0.0)
            for idx, nn in enumerate(sorted_nns[test_sample, :]):
                if nn == idxs_test[test_sample]:
                    continue
                votes_per_label[np.int32(self.labels[np.int32(nn)])] += weight[idx]
            pred_labels[test_sample, :] = np.asarray(votes_per_label.values())

        # Get maximal predicted label for each test sample and return acc
        pred_labels = pred_labels.argmax(axis=1)
        acc = eval.accuracy_score(y_pred=pred_labels, y_true=self.labels[idxs_test])
        return acc

    def ten_fold_cv_train(self, n):
        """
        Run experiment 10 times reporting mean and std
        :param n: number of neighbours to calculate prediction
        :return:
        """
        acc = []
        for i in range(10):
            acc.append(self.nn_classifier(n))
        return np.asarray(acc).mean(), np.asarray(acc).std()
