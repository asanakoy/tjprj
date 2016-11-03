import numpy as np
import multiprocessing as mp
import ctypes

CATEGORIES = ['basketball_layup',
              'bowling',
              'clean_and_jerk',
              'discus_throw',
              'diving_platform_10m',
              'diving_springboard_3m',
              'hammer_throw',
              'high_jump',
              'javelin_throw',
              'long_jump',
              'pole_vault',
              'shot_put',
              'snatch',
              'tennis_serve',
              'triple_jump',
              'vault']

# CATEGORIES = ['snatch', 'vault']


class BatchWorker(mp.Process):
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, images_shared_arr, images_arr_shape, labels_shared_arr, queue):
        super(BatchWorker, self).__init__()
        self._queue = queue
        self.images = np.frombuffer(images_shared_arr.get_obj()).reshape(images_arr_shape)
        self.labels = np.frombuffer(labels_shared_arr.get_obj())

    def run(self):
        pass


class BatchManager(object):
    def __init__(self, batch_loaders, batch_size, image_shape, random_shuffle=True,
                 random_seed=None):
        if len(batch_loaders) != len(CATEGORIES):
            raise ValueError
        if len(image_shape) != 2:
            raise ValueError('Image shape must be a pair (h, w)')

        self.num_categories = len(CATEGORIES)
        self.batch_loaders = batch_loaders
        self.batch_size = batch_size
        self._cur_batch_num = 0
        self.random_shuffle = random_shuffle
        self.random_state = np.random.RandomState(random_seed)

        # TODO: make asynchronous, for this we need to create batchloaders in a separate process
        # self._lock = mp.Lock()
        images_arr_shape = (self.num_categories * self.batch_size, image_shape[0], image_shape[1], 3)
        # images_shared_arr = mp.Array(ctypes.c_float, np.prod(images_arr_shape))
        # self.images = np.frombuffer(images_shared_arr.get_obj()).reshape(images_arr_shape)
        #
        num_labels = self.num_categories * self.batch_size
        # labels_shared_arr = mp.Array(ctypes.c_int32, num_labels)
        # self.labels = np.frombuffer(labels_shared_arr.get_obj())

        self.images = np.zeros(images_arr_shape, dtype=np.float32)
        self.labels = np.zeros(num_labels, dtype=np.int32)

    def __fetch_category_batches(self):
        assert self._cur_batch_num == 0
        offset = 0
        for cat_name, batch_loader in self.batch_loaders.iteritems():
            images, labels = batch_loader.get_next_batch(self.batch_size)
            self.images[offset: offset + self.batch_size, ...] = images.transpose((0, 2, 3, 1))
            self.labels[offset: offset + self.batch_size] = labels
            offset += self.batch_size
        if self.random_shuffle:
            perm = self.random_state.permutation(len(self.labels))
            self.images = self.images[perm]
            self.labels = self.labels[perm]

    def fill_feed_dict(self, phase='test'):
        """Fills the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        if phase not in ['train', 'test']:
            raise ValueError('phase must be "train" or "test"')
        if phase == 'train':
            keep_prob = 0.5
            is_phase_train = True
        else:
            keep_prob = 1.0
            is_phase_train = False

        if self._cur_batch_num == self.num_categories:
            self._cur_batch_num = 0
            self.__fetch_category_batches()

        pos_begin = self._cur_batch_num * self.batch_size
        images_feed = self.images[pos_begin:pos_begin + self.batch_size]
        labels_feed = self.labels[pos_begin:pos_begin + self.batch_size]
        self._cur_batch_num += 1

        feed_dict = {
            'input/x:0': images_feed,
            'input/y_gt:0': labels_feed,
            'fc6/keep_prob_pl:0': keep_prob,
            'fc7/keep_prob_pl:0': keep_prob,
            'input/is_phase_train:0': is_phase_train
        }
        return feed_dict

    def cleanup(self):
        for batch_laoder in self.batch_loaders():
            batch_laoder.cleanup_workers()
