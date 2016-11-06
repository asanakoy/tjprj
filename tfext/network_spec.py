# Original work Copyright 2015 The TensorFlow Authors. Licensed under the Apache License v2.0 http://www.apache.org/licenses/LICENSE-2.0
# Modified work Copyright (c) 2016 Artsiom Sanakoyeu
"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def placeholder_inputs(batch_size, image_shape):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,) + image_shape)
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss_value = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss_value


def soft_xe(x, y, alpha, num_classes_in_batch):

    # Compute each cluster representative (mean)
    clique_examples = tf.dynamic_partition(x, y, num_classes_in_batch)
    r = tf.pack([tf.reduce_mean(x_aux, 0) for x_aux in clique_examples])

    # Compute distance from all points to all clusters representatives

    dr = tf.squared_difference(tf.expand_dims(x, 1), r)
    dr = tf.reduce_sum(dr, 2)

    # Select distances of examples to their own centroid
    assignment_clique = tf.to_float(comparison_mask(y, np.arange(num_classes_in_batch, dtype=np.int32)))
    d_xi_ri = tf.reduce_sum(dr * assignment_clique, 1)

    # Compute std of intra-cluster distances
    N = tf.shape(x)[0]
    sigma = tf.reduce_sum(d_xi_ri) / tf.to_float(N)
    std_normalizer = -1.0 / (2.0 * sigma ** 2.0)

    # Compute numerator
    numerator = tf.exp(std_normalizer * d_xi_ri - alpha)

    # Compute denominator
    d_xi_rj = tf.exp(std_normalizer * dr)
    denominator = tf.reduce_sum(d_xi_rj, 1)

    # Compute example losses and total loss
    epsilon = 1e-8
    losses = tf.nn.relu(-tf.log(numerator / (denominator + epsilon) + epsilon))
    soft_xe = tf.reduce_mean(losses)

    return soft_xe



def comparison_mask(a_labels, b_labels):
    return tf.equal(tf.expand_dims(a_labels, 1),
                    tf.expand_dims(b_labels, 0))


def training_stl(net, loss, base_lr=None):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    tf.scalar_summary(loss.op.name, loss)
    # WARNING: initial_accumulator_value in caffe's AdaGrad is probably 0.0
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    assert len(update_ops) > 0
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdagradOptimizer(base_lr, initial_accumulator_value=0.0001)
        op = optimizer.minimize(loss=loss, global_step=net.global_iter_counter)
    return op

def training_sgd(net, loss, base_lr=None):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(base_lr)
    op = optimizer.minimize(loss=loss)
    return op

def training_stl_eval(net, loss, base_lr=None):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    tf.scalar_summary(loss.op.name, loss)
    # WARNING: initial_accumulator_value in caffe's AdaGrad is probably 0.0
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    assert len(update_ops) > 0
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdagradOptimizer(learning_rate=base_lr, initial_accumulator_value=0.0001)
        # optimizer = tf.train.GradientDescentOptimizer(base_lr)
        op = optimizer.minimize(loss=loss)
    return op


def training_sgd(net, loss, base_lr=None):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(base_lr)
    op = optimizer.minimize(loss=loss)
    return op

def training_stl_eval(net, loss, base_lr=None):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    tf.scalar_summary(loss.op.name, loss)
    # WARNING: initial_accumulator_value in caffe's AdaGrad is probably 0.0
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    assert len(update_ops) > 0
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdagradOptimizer(learning_rate=base_lr, initial_accumulator_value=0.0001)
        # optimizer = tf.train.GradientDescentOptimizer(base_lr)
        op = optimizer.minimize(loss=loss)
    return op


def training(net, loss_op, base_lr=None, fc_lr_mult=1.0, conv_lr_mult=1.0, **params):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss_op: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    tf.scalar_summary(loss_op.op.name, loss_op)
    # WARNING: initial_accumulator_value in caffe's AdaGrad is probably 0.0
    conv_optimizer = tf.train.AdagradOptimizer(base_lr * conv_lr_mult,
                                               initial_accumulator_value=0.0001)
    fc_optimizer = tf.train.AdagradOptimizer(base_lr * fc_lr_mult,
                                             initial_accumulator_value=0.0001)

    print('Conv LR: {}, FC LR: {}'.format(base_lr * conv_lr_mult, base_lr * fc_lr_mult))

    assert len(net.trainable_vars) == len(tf.trainable_variables())

    conv_vars = [val for (key, val) in net.trainable_vars.iteritems() if key.startswith('conv')]
    fc_vars = [val for (key, val) in net.trainable_vars.iteritems() if key.startswith('fc')]

    grads = tf.gradients(loss_op, conv_vars + fc_vars)
    conv_grads = grads[:len(conv_vars)]
    fc_grads = grads[len(conv_vars):]
    assert len(conv_grads) + len(fc_grads) == len(net.trainable_vars)

    conv_tran_op = conv_optimizer.apply_gradients(zip(conv_grads, conv_vars))
    fc_tran_op = fc_optimizer.apply_gradients(zip(fc_grads, fc_vars),
                                              global_step=net.global_iter_counter)
    return tf.group(conv_tran_op, fc_tran_op)


def training_convnet(net, loss_op, fc_lr, conv_lr):

    conv_optimizer = tf.train.AdagradOptimizer(conv_lr,
                                               initial_accumulator_value=0.0001)
    fc_optimizer = tf.train.AdagradOptimizer(fc_lr,
                                             initial_accumulator_value=0.0001)

    print('Conv LR: {}, FC LR: {}'.format(conv_lr, fc_lr))

    conv_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv')
    fc_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')

    grads = tf.gradients(loss_op, conv_vars + fc_vars)
    conv_grads = grads[:len(conv_vars)]
    fc_grads = grads[len(conv_vars):]
    assert len(conv_grads) + len(fc_grads) == len(conv_vars) + len(fc_vars)
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        conv_tran_op = conv_optimizer.apply_gradients(zip(conv_grads, conv_vars))
        fc_tran_op = fc_optimizer.apply_gradients(zip(fc_grads, fc_vars),
                                                  global_step=net.global_iter_counter)
    return tf.group(conv_tran_op, fc_tran_op)


def correct_classified_top1(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
