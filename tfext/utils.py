# Artsiom Sanakoyeu, 2016
from __future__ import division
import numpy as np
from tensorflow.core.framework import summary_pb2


def fill_feed_dict(net, batch_loader, batch_size=128, phase='test'):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      batch_loader: BatchLoader, that provides batches of the data
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

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


    images_feed, labels_feed = batch_loader.get_next_batch(batch_size)
    # TODO: remove after I fix BatchLoader. Presently BatchLoader outputs CxHxW BGR images.
    images_feed = images_feed.transpose((0, 2, 3, 1))

    feed_dict = {
        net.x: images_feed,
        net.y_gt: labels_feed,
        net.fc6_keep_prob: keep_prob,
        net.fc7_keep_prob: keep_prob,
        'input/is_phase_train:0': is_phase_train
    }
    return feed_dict


def fill_feed_dict_convnet(net, batch_loader, batch_size=128, phase='test'):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      batch_loader: BatchLoader, that provides batches of the data
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

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

    images_feed, labels_feed = batch_loader.get_next_batch(batch_size)
    # TODO: remove after I fix BatchLoader. Presently BatchLoader outputs CxHxW images.
    images_feed = images_feed.transpose((0, 2, 3, 1))

    feed_dict = {
        net.x: images_feed,
        net.y_gt: labels_feed,
        net.dropout_keep_prob: keep_prob,
        'input/is_phase_train:0': is_phase_train
    }
    return feed_dict


def fill_feed_dict_magnet(net, batch_loader, batch_size=128, max_cliques_per_batch=8, phase='test'):

    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      batch_loader: BatchLoader, that provides batches of the data
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

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

    images_feed, labels_feed = batch_loader.get_next_batch(batch_size)

    unq_y = np.unique(labels_feed)
    labels_feed_aux = np.where(np.tile(unq_y, (labels_feed.shape[0], 1)) == labels_feed.reshape(-1, 1))[1]

    idxs_samples = np.where(labels_feed_aux < max_cliques_per_batch)[0]
    assert idxs_samples.shape[0] > max_cliques_per_batch
    images_feed = images_feed[idxs_samples, :, :, :]
    labels_feed_aux = labels_feed_aux[idxs_samples]
    # TODO: remove after I fix BatchLoader. Presently BatchLoader outputs CxHxW images.
    images_feed = images_feed.transpose((0, 2, 3, 1))

    feed_dict = {
        net.x: images_feed,
        net.y_gt: labels_feed_aux,
        net.fc6_keep_prob: keep_prob,
        # net.fc7_keep_prob: keep_prob,
        'input/is_phase_train:0': is_phase_train
    }
    return feed_dict


def fill_feed_dict_stl(net, batch_loader, batch_size=128, phase='test'):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      batch_loader: BatchLoader, that provides batches of the data
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

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

    images_feed, labels_feed = batch_loader.get_next_batch(batch_size)


    # TODO: remove after I fix BatchLoader. Presently BatchLoader outputs CxHxW images.
    images_feed = images_feed.transpose((0, 2, 3, 1))
    unq_y = np.unique(labels_feed)
    labels_feed_aux = np.where(np.tile(unq_y, (labels_feed.shape[0], 1)) == labels_feed.reshape(-1, 1))[1]

    feed_dict = {
        net.x: images_feed,
        net.y_gt: labels_feed_aux,
        net.dropout_keep_prob: keep_prob,
        'input/is_phase_train:0': is_phase_train
    }
    return feed_dict


def calc_acuracy(net,
                 sess,
                 eval_correct,
                 batch_loader,
                 batch_size=128,
                 num_images=None):
    """Runs one correct_classified_top1 against the full epoch of data.

    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      batch_loader: BatchLoader, that provides batches of the data
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    if num_images is None:
        steps_num = len(batch_loader.indexlist) // batch_size
    else:
        assert num_images <= len(batch_loader.indexlist)
        steps_num = num_images // batch_size


    # WARNING: be careful! We change the internal pointer of the BatchLoader.
    old_batch_loader_position = batch_loader._cur

    batch_loader._cur = 0
    num_examples = steps_num * batch_size
    for step in xrange(steps_num):
        feed_dict = fill_feed_dict(net,
                                   batch_loader,
                                   batch_size=batch_size,
                                   phase='test')
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    # restore the internal pointer
    batch_loader._cur = old_batch_loader_position

    accuracy = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d Accuracy @ 1: %0.04f' %

          (num_examples, true_count, accuracy))


def create_sumamry(tag, value):
    """
    Create a summary for logging via tf.train.SummaryWriter
    """
    x = summary_pb2.Summary.Value(tag=tag, simple_value=value)
    return summary_pb2.Summary(value=[x])
