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


def soft_xe(x, y, alpha, num_classes_in_batch, sess=None):

    """
    This function computes the soft cross entropy loss.
    :param x: representation of samples in batch (batch_size, ndims)
    :param y: labels for each sample [0, k]
    :param alpha: gap between positive and negative classes
    :param num_classes_in_batch: number of classes in the batch
    :param sess: session for debugging purposes
    :return:
    """
    # Compute each cluster representative (centroid)
    clique_examples = tf.dynamic_partition(x, y, num_classes_in_batch)
    idxs_centroid = []
    for clique in clique_examples:
        sqdif = tf.reduce_sum(tf.squared_difference(tf.expand_dims(clique, 1), clique), 2)
        tf.Print(sqdif, [sqdif], 'This is sqdif')
        idxs_centroid.append(tf.to_int32(tf.argmin(tf.reduce_sum(sqdif, 0), 0)))
    r = tf.pack([x_aux[idxs_centroid[_i]] for _i, x_aux in enumerate(clique_examples)])

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
    return tf.reduce_mean(losses)



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
    optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr, momentum=0.9)
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
                                               initial_accumulator_value=0.0000001)
    fc_optimizer = tf.train.AdagradOptimizer(base_lr * fc_lr_mult,
                                             initial_accumulator_value=0.0000001)

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

def training_fix_upto_fc6(net, loss_op, lower_lr, upper_lr):

    conv_optimizer = tf.train.AdagradOptimizer(learning_rate=lower_lr, initial_accumulator_value=0.00001)
    fc_optimizer = tf.train.AdagradOptimizer(learning_rate=upper_lr, initial_accumulator_value=0.00001)

    print('Lower(incl fc6) LR: {}, Upper LR: {}'.format(lower_lr, upper_lr))

    fixed_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv') +\
                net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc6')
    variable_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')
    variable_vars = [tensor for tensor in variable_vars if not str(tensor.name).startswith('fc6')]

    grads = tf.gradients(loss_op, fixed_vars + variable_vars)
    conv_grads = grads[:len(fixed_vars)]
    fc_grads = grads[len(fixed_vars):]
    assert len(conv_grads) + len(fc_grads) == len(fixed_vars) + len(variable_vars)
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        conv_tran_op = conv_optimizer.apply_gradients(zip(conv_grads, fixed_vars))
        fc_tran_op = fc_optimizer.apply_gradients(zip(fc_grads, variable_vars),
                                                  global_step=net.global_iter_counter)
    return tf.group(conv_tran_op, fc_tran_op)

def training_stl_freeze_conv9(net, loss_op, base_lr):

    lower_optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr, momentum=0.9)
    upper_optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr, momentum=0.9)

    print('Lower LR: {}, Upper LR: {}'.format(base_lr, base_lr))

    fixed_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv')
    variable_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')

    # Filter tensors into fixed and variable
    fixed_vars_aux = []
    for tensor in fixed_vars:
        tensor_name = str(tensor.name)
        tensor_num = int(filter(str.isdigit, tensor_name)[0])
        if tensor_num <= 9:
            fixed_vars_aux.append(tensor)
        else:
            variable_vars.append(tensor)
    fixed_vars = fixed_vars_aux

    grads = tf.gradients(loss_op, fixed_vars + variable_vars)
    lower_grads = grads[:len(fixed_vars)]
    upper_grads = grads[len(fixed_vars):]
    assert len(lower_grads) + len(upper_grads) == len(fixed_vars) + len(variable_vars)
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        conv_tran_op = lower_optimizer.apply_gradients(zip(lower_grads, fixed_vars))
        fc_tran_op = upper_optimizer.apply_gradients(zip(upper_grads, variable_vars),
                                                  global_step=net.global_iter_counter)
    return tf.group(conv_tran_op, fc_tran_op)



def training_warmup_stl(net, loss_op, lower_lr, upper_lr):

    conv_optimizer = tf.train.AdagradOptimizer(lower_lr,
                                               initial_accumulator_value=0.0001)
    fc_optimizer = tf.train.AdagradOptimizer(upper_lr,
                                             initial_accumulator_value=0.0001)

    print('Lower(incl fc6) LR: {}, Upper LR: {}'.format(lower_lr, upper_lr))

    fixed_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv')
    variable_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')

    grads = tf.gradients(loss_op, fixed_vars + variable_vars)
    conv_grads = grads[:len(fixed_vars)]
    fc_grads = grads[len(fixed_vars):]
    assert len(conv_grads) + len(fc_grads) == len(fixed_vars) + len(variable_vars)
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        conv_tran_op = conv_optimizer.apply_gradients(zip(conv_grads, fixed_vars))
        fc_tran_op = fc_optimizer.apply_gradients(zip(fc_grads, variable_vars),
                                                  global_step=net.global_iter_counter)
    return tf.group(conv_tran_op, fc_tran_op)


def training_warmup_stl_exemplarcnn(net, loss_op, lower_lr, upper_lr):

    lower_optimizer = tf.train.MomentumOptimizer(lower_lr, momentum=0.9)
    upper_optimizer = tf.train.MomentumOptimizer(upper_lr, momentum=0.9)

    print('Lower(incl fc6) LR: {}, Upper LR: {}'.format(lower_lr, upper_lr))

    lower_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'input/conv')
    upper_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')

    # Filter out fc6 which is fixed
    upper_vars_aux = []
    for tensor in upper_vars:
        tensor_name = str(tensor.name)
        tensor_num = int(filter(str.isdigit, tensor_name)[0])
        if tensor_num == 6:
            lower_vars.append(tensor)
        else:
            upper_vars_aux.append(tensor)
    upper_vars = upper_vars_aux

    grads = tf.gradients(loss_op, lower_vars + upper_vars)
    lower_grads = grads[:len(lower_vars)]
    upper_grads = grads[len(lower_vars):]
    assert len(lower_grads) + len(upper_grads) == len(lower_vars) + len(upper_vars)
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        conv_tran_op = lower_optimizer.apply_gradients(zip(lower_grads, lower_vars))
        fc_tran_op = upper_optimizer.apply_gradients(zip(upper_grads, upper_vars),
                                                  global_step=net.global_iter_counter)
    return tf.group(conv_tran_op, fc_tran_op)

def training_warmup_stl(net, loss_op, lower_lr, upper_lr):

    conv_optimizer = tf.train.AdagradOptimizer(lower_lr,
                                               initial_accumulator_value=0.0001)
    fc_optimizer = tf.train.AdagradOptimizer(upper_lr,
                                             initial_accumulator_value=0.0001)

    print('Lower(incl fc6) LR: {}, Upper LR: {}'.format(lower_lr, upper_lr))

    fixed_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv')
    variable_vars = net.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')

    grads = tf.gradients(loss_op, fixed_vars + variable_vars)
    conv_grads = grads[:len(fixed_vars)]
    fc_grads = grads[len(fixed_vars):]
    assert len(conv_grads) + len(fc_grads) == len(fixed_vars) + len(variable_vars)
    update_ops = net.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        conv_tran_op = conv_optimizer.apply_gradients(zip(conv_grads, fixed_vars))
        fc_tran_op = fc_optimizer.apply_gradients(zip(fc_grads, variable_vars),
                                                  global_step=net.global_iter_counter)
    return tf.group(conv_tran_op, fc_tran_op)




def training_convnet(net, loss_op, fc_lr, conv_lr, optimizer_type='adagrad',
                     trace_gradients=False):
    with net.graph.as_default():
        print('Creating optimizer {}'.format(optimizer_type))
        if optimizer_type == 'adagrad':
            conv_optimizer = tf.train.AdagradOptimizer(conv_lr,
                                                       initial_accumulator_value=0.0001)
            fc_optimizer = tf.train.AdagradOptimizer(fc_lr,
                                                     initial_accumulator_value=0.0001)
        elif optimizer_type == 'sgd':
            conv_optimizer = tf.train.GradientDescentOptimizer(conv_lr)
            fc_optimizer = tf.train.GradientDescentOptimizer(fc_lr)
        elif optimizer_type == 'momentum':
            conv_optimizer = tf.train.MomentumOptimizer(conv_lr, momentum=0.9)
            fc_optimizer = tf.train.MomentumOptimizer(fc_lr, momentum=0.9)
        else:
            raise ValueError('Unknown optimizer type {}'.format(optimizer_type))

        print('Conv LR: {}, FC LR: {}'.format(conv_lr, fc_lr))

        conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv')
        fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')

        assert len(conv_vars) + len(fc_vars) == \
            len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)),\
            'You dont train all the variables'

        grads = tf.gradients(loss_op, conv_vars + fc_vars)
        conv_grads = grads[:len(conv_vars)]
        fc_grads = grads[len(conv_vars):]
        assert len(conv_grads) == len(conv_vars)
        assert len(fc_grads) == len(fc_vars)

        with tf.name_scope('grad_norms'):
            for v, grad in zip(conv_vars + fc_vars, grads):
                if grad is not None:
                    grad_norm_op = tf.nn.l2_loss(grad, name=format(v.name[:-2]))
                    tf.add_to_collection('grads', grad_norm_op)
                    if trace_gradients:
                        tf.scalar_summary(grad_norm_op.name, grad_norm_op)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            conv_tran_op = conv_optimizer.apply_gradients(zip(conv_grads, conv_vars), name='conv_train_op')
        fc_tran_op = fc_optimizer.apply_gradients(zip(fc_grads, fc_vars), global_step=net.global_iter_counter, name='fc_train_op')
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
