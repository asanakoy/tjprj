################################################################################
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
# Artsiom Sanakoyeu, 2016
################################################################################
import numpy as np
import os
import tensorflow as tf


class Alexnet(object):
    """
    Net description
    (self.feed('data')
            .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            .lrn(2, 2e-05, 0.75, name='norm1')
            .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            .conv(5, 5, 256, 1, 1, group=2, name='conv2')
            .lrn(2, 2e-05, 0.75, name='norm2')
            .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            .conv(3, 3, 384, 1, 1, name='conv3')
            .conv(3, 3, 384, 1, 1, group=2, name='conv4')
            .conv(3, 3, 256, 1, 1, group=2, name='conv5')
            .fc(4096, name='fc6')
            .fc(4096, name='fc7')
            .fc(num_classes, relu=False, name='fc8')
            .softmax(name='prob'))

    WARNING! You should feed images in HxWxC BGR format!
    """

    def __init__(self, init_model=None, num_classes=1000,
                 im_shape=(227, 227, 3), device_id='/gpu:0', num_layers_to_init=8, **params):
        """
        :param init_model: dict containing network weights,
                           or a string with path to .np file with the dict,
                           if is None then init using random weights and biases
        :return:
        """
        self.input_shape = im_shape
        self.num_classes = num_classes
        self.device_id = device_id
        self.num_layers_to_init = num_layers_to_init
        tr_vars = dict()

        if len(self.input_shape) == 2:
            self.input_shape += (3,)

        assert len(self.input_shape) == 3
        if self.num_layers_to_init > 8 or self.num_layers_to_init < 0:
            raise ValueError('Number of layer to init must be in [0, 8] ({} provided)'.
                             format(self.num_layers_to_init))

        if init_model is None:
            net_data = None
        elif isinstance(init_model, basestring):
            if not os.path.exists(init_model):
                raise IOError('Net Weights file not found: {}'.format(init_model))
            print 'Loading Net Weights from: {}'.format(init_model)
            net_data = np.load(init_model).item()

        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, (None,) + self.input_shape, name='x')
            self.y_gt = tf.placeholder(tf.int32, shape=(None,), name='y_gt')

        with tf.device(self.device_id):
            layer_index = 0
            # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            with tf.variable_scope('conv1'):
                kernel_height = 11
                kernel_width = 11
                kernels_num = 96
                group = 1
                num_input_channels = int(self.x.get_shape()[3])
                s_h = 4  # stride by the H dimension (height)
                s_w = 4  # stride by the W dimension (width)
                tr_vars['conv1w'], tr_vars['conv1b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels / group, kernels_num)
                layer_index += 1
                conv1_in = Alexnet.conv(self.x, tr_vars['conv1w'], tr_vars['conv1b'],
                                        kernel_height, kernel_width,
                                        kernels_num, s_h, s_w, padding="SAME", group=group)
                conv1 = tf.nn.relu(conv1_in)

                # lrn1
                # lrn(2, 2e-05, 0.75, name='norm1')
                radius = 2
                alpha = 2e-05
                beta = 0.75
                bias = 1.0
                lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias, name='lrn')

                # maxpool1
                # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
                kernel_height = 3
                kernel_width = 3
                s_h = 2
                s_w = 2
                padding = 'VALID'
                maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, kernel_height, kernel_width, 1],
                                          strides=[1, s_h, s_w, 1], padding=padding,
                                          name='maxpool')

            # conv(5, 5, 256, 1, 1, group=2, name='conv2')
            with tf.variable_scope('conv2'):
                kernel_height = 5
                kernel_width = 5
                kernels_num = 256
                num_input_channels = int(maxpool1.get_shape()[3])
                s_h = 1
                s_w = 1
                group = 2
                tr_vars['conv2w'], tr_vars['conv2b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels / group, kernels_num)
                layer_index += 1
                conv2_in = Alexnet.conv(maxpool1, tr_vars['conv2w'], tr_vars['conv2b'],
                                        kernel_height, kernel_width,
                                        kernels_num, s_h, s_w, padding="SAME", group=group)
                conv2 = tf.nn.relu(conv2_in, name='relu')

                # lrn2
                # lrn(2, 2e-05, 0.75, name='norm2')
                radius = 2
                alpha = 2e-05
                beta = 0.75
                bias = 1.0
                lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

                # maxpool2
                # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
                kernel_height = 3
                kernel_width = 3
                s_h = 2
                s_w = 2
                padding = 'VALID'
                maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, kernel_height, kernel_width, 1],
                                          strides=[1, s_h, s_w, 1], padding=padding)

            # conv(3, 3, 384, 1, 1, name='conv3')
            with tf.variable_scope('conv3'):
                kernel_height = 3
                kernel_width = 3
                kernels_num = 384
                num_input_channels = int(maxpool2.get_shape()[3])
                s_h = 1
                s_w = 1
                group = 1
                tr_vars['conv3w'], tr_vars['conv3b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels / group, kernels_num)
                layer_index += 1
                conv3_in = Alexnet.conv(maxpool2, tr_vars['conv3w'], tr_vars['conv3b'],
                                        kernel_height, kernel_width,
                                        kernels_num, s_h, s_w, padding="SAME", group=group)

                conv3 = tf.nn.relu(conv3_in, 'relu')

            # conv(3, 3, 384, 1, 1, group=2, name='conv4')
            with tf.variable_scope('conv4'):
                kernel_height = 3
                kernel_width = 3
                kernels_num = 384
                num_input_channels = int(conv3.get_shape()[3])
                s_h = 1
                s_w = 1
                group = 2
                tr_vars['conv4w'], tr_vars['conv4b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels / group, kernels_num)
                layer_index += 1
                conv4_in = Alexnet.conv(conv3, tr_vars['conv4w'], tr_vars['conv4b'],
                                        kernel_height, kernel_width,
                                        kernels_num, s_h, s_w, padding="SAME", group=group)
                conv4 = tf.nn.relu(conv4_in, name='relu')

            # conv(3, 3, 256, 1, 1, group=2, name='conv5')
            with tf.variable_scope('conv5'):
                kernel_height = 3
                kernel_width = 3
                kernels_num = 256
                num_input_channels = int(conv4.get_shape()[3])
                s_h = 1
                s_w = 1
                group = 2
                tr_vars['conv5w'], tr_vars['conv5b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels / group, kernels_num)
                layer_index += 1
                self.conv5 = Alexnet.conv(conv4, tr_vars['conv5w'], tr_vars['conv5b'],
                                          kernel_height, kernel_width,
                                          kernels_num, s_h, s_w, padding="SAME", group=group)
                self.conv5_relu = tf.nn.relu(self.conv5, name='relu')

                # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
                kernel_height = 3
                kernel_width = 3
                s_h = 2
                s_w = 2
                padding = 'VALID'
                self.maxpool5 = tf.nn.max_pool(self.conv5_relu,
                                               ksize=[1, kernel_height, kernel_width, 1],
                                               strides=[1, s_h, s_w, 1], padding=padding,
                                               name='maxpool')

            # fc(4096, name='fc6')
            with tf.variable_scope('fc6'):
                num_inputs = int(np.prod(self.maxpool5.get_shape()[1:]))
                num_outputs = 4096
                tr_vars['fc6w'], tr_vars['fc6b'] = \
                    self.get_fc_weights(layer_index, net_data, num_inputs, num_outputs)
                layer_index += 1
                self.fc6 = tf.add(tf.matmul(
                    tf.reshape(self.maxpool5,
                               [-1, int(np.prod(self.maxpool5.get_shape()[1:]))]
                               ),
                    tr_vars['fc6w']),
                    tr_vars['fc6b'], name='fc')
                self.fc6_relu = tf.nn.relu(self.fc6, name='relu')

                self.fc6_keep_prob = tf.placeholder_with_default(1.0, tuple(),
                                                                 name='keep_prob_pl')
                fc6_dropout = tf.nn.dropout(self.fc6_relu, self.fc6_keep_prob, name='dropout')

            # fc(4096, name='fc7')
            with tf.variable_scope('fc7'):
                num_inputs = int(fc6_dropout.get_shape()[1])
                num_outputs = 4096
                tr_vars['fc7w'], tr_vars['fc7b'] = \
                    self.get_fc_weights(layer_index, net_data, num_inputs, num_outputs)
                layer_index += 1
                self.fc7 = tf.add(tf.matmul(fc6_dropout, tr_vars['fc7w']), tr_vars['fc7b'],
                                  name='fc')
                self.fc7_relu = tf.nn.relu(self.fc7, name='relu')

                self.fc7_keep_prob = tf.placeholder_with_default(1.0, tuple(),
                                                                 name='keep_prob_pl')
                fc7_dropout = tf.nn.dropout(self.fc7_relu, self.fc7_keep_prob, name='dropout')

            # fc(num_classes, relu=False, name='fc8')
            with tf.variable_scope('fc8'):
                num_inputs = int(fc7_dropout.get_shape()[1])
                num_outputs = self.num_classes
                tr_vars['fc8w'], tr_vars['fc8b'] = \
                    self.get_fc_weights(layer_index, net_data, num_inputs, num_outputs)
                layer_index += 1
                self.fc8 = tf.add(tf.matmul(fc7_dropout, tr_vars['fc8w']), tr_vars['fc8b'],
                                  name='fc')
                assert self.fc8.get_shape()[1] == self.num_classes, \
                    '{} != {}'.format(self.fc8.get_shape()[1], self.num_classes)

            with tf.variable_scope('output'):
                self.prob = tf.nn.softmax(self.fc8, name='prob')

        self.trainable_vars = tr_vars
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                                     allow_soft_placement=True))

    def get_conv_weights(self, layer_index, net_data, kernel_height, kernel_width,
                         num_input_channels, kernels_num):
        layer_names = ['conv{}'.format(i) for i in xrange(1, 6)] + \
                      ['fc{}'.format(i) for i in xrange(6, 9)]
        wights_std = [0.01] * 5 + [0.005, 0.005, 0.01]
        bias_init_values = [0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0]

        l_name = layer_names[layer_index]
        if net_data is not None and layer_index < self.num_layers_to_init:
            assert net_data[l_name][0].shape == (kernel_height, kernel_width,
                                                 num_input_channels,
                                                 kernels_num)
            assert net_data[l_name][1].shape == (kernels_num,)

        if layer_index >= self.num_layers_to_init or net_data is None:
            print 'Initializing {} with random'.format(l_name)
            w = self.random_weight_variable((kernel_height, kernel_width,
                                             num_input_channels,
                                             kernels_num),
                                            stddev=wights_std[layer_index])
            b = self.random_bias_variable((kernels_num,), value=bias_init_values[layer_index])
        else:
            w = tf.Variable(net_data[l_name][0], name='weight')
            b = tf.Variable(net_data[l_name][1], name='bias')
        return w, b

    def get_fc_weights(self, layer_index, net_data, num_inputs, num_outputs):
        layer_names = ['conv{}'.format(i) for i in xrange(1, 6)] + \
                      ['fc{}'.format(i) for i in xrange(6, 9)]
        wights_std = [0.01] * 5 + [0.005, 0.005, 0.01]
        bias_init_values = [0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0]
        l_name = layer_names[layer_index]
        if net_data is not None and layer_index < self.num_layers_to_init:
            assert net_data[l_name][0].shape == (num_inputs, num_outputs)
            assert net_data[l_name][1].shape == (num_outputs,)

        if layer_index >= self.num_layers_to_init or net_data is None:
            print 'Initializing {} with random'.format(l_name)
            w = self.random_weight_variable((num_inputs, num_outputs),
                                            stddev=wights_std[layer_index])
            b = self.random_bias_variable((num_outputs,), value=bias_init_values[layer_index])
        else:
            w = tf.Variable(net_data[l_name][0], name='weight')
            b = tf.Variable(net_data[l_name][1], name='bias')
        return w, b

    @staticmethod
    def random_weight_variable(shape, stddev=0.01):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name='weight')

    @staticmethod
    def random_bias_variable(shape, value=0.1):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial, name='bias')

    @staticmethod
    def conv(input, kernel, biases, kernel_height, kernel_width,
             kernels_num, s_h, s_w, padding="VALID", group=1):
        """
        From https://github.com/ethereon/caffe-tensorflow
        """
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert kernels_num % group == 0

        def convolve(inp, w, name=None):
            return tf.nn.conv2d(inp, w, [1, s_h, s_w, 1], padding=padding, name=name)

        if group == 1:
            conv = convolve(input, kernel, name='conv')
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        return tf.reshape(tf.nn.bias_add(conv, biases),
                          [-1] + conv.get_shape().as_list()[1:], name='conv')
