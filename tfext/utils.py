# Artsiom Sanakoyeu, 2016
from __future__ import division
import numpy as np

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
    assert phase in ['train', 'test']
    keep_prob = 1.0 if phase == 'test' else 0.5

    images_feed, labels_feed = batch_loader.get_next_batch(batch_size)
    # TODO: remove after I fix BatchLoader. Presently BatchLoader outputs CxHxW images.
    images_feed = images_feed.transpose((0, 2, 3, 1))

    feed_dict = {
        net.x: images_feed,
        net.y_gt: labels_feed,
        net.fc6_keep_prob: keep_prob,
        net.fc7_keep_prob: keep_prob
    }
    return feed_dict

def fill_feed_dict_magnet(net, mu_ph, unique_mu_ph, sigma_ph, batch_loader, centroider, batch_size=128, phase='test'):
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
    assert phase in ['train', 'test']
    keep_prob = 1.0 if phase == 'test' else 0.5

    images_feed, labels_feed = batch_loader.get_next_batch(batch_size)
    mu_feed = centroider.get_nearest_mu(labels_feed)
    _, unique_labels = np.unique(labels_feed, return_index=True)
    unique_mu_feed = mu_feed[unique_labels]
    sigma_feed = centroider.get_sigma(labels_feed)
    # TODO: remove after I fix BatchLoader. Presently BatchLoader outputs CxHxW images.
    images_feed = images_feed.transpose((0, 2, 3, 1))

    feed_dict = {
        net.x: images_feed,
        net.y_gt: labels_feed,
        mu_ph: mu_feed,
        unique_mu_ph: unique_mu_feed,
        sigma_ph: sigma_feed,
        net.fc6_keep_prob: keep_prob,
        net.fc7_keep_prob: keep_prob
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