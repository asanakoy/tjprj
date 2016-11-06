################################################################################
# CaffeNet
#
# Artsiom Sanakoyeu, 2016
################################################################################
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from tfext.alexnet import Alexnet


class CaffeNet(Alexnet):
    """
    Net description
         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .lrn(2, 2e-05, 0.75, name='norm1')
         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .lrn(2, 2e-05, 0.75, name='norm2')
         .conv(3, 3, 384, 1, 1, name='conv3')
         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(1000, relu=False, name='fc8')
         .softmax(name='prob'))

    WARNING! You should feed images in HxWxC BGR format!
    """

    def __init__(self, init_model=None, num_classes=1,
                 im_shape=(227, 227, 3), device_id='/gpu:0', num_layers_to_init=8,
                 random_init_type=Alexnet.RandomInitType.GAUSSIAN, use_batch_norm=False,
                 gpu_memory_fraction=None, **params):

        """
         Args:
          init_model: dict containing network weights, or a string with path to .np file with the dict,
            if is None then init using random weights and biases
          num_classes: number of output classes
          gpu_memory_fraction: Fraction on the max GPU memory to allocate for process needs.
            Allow auto growth if None (can take up to the totality of the memory).
        :return:
        """
        self.input_shape = im_shape
        self.num_classes = num_classes
        self.device_id = device_id
        self.num_layers_to_init = num_layers_to_init
        self.random_init_type = random_init_type
        self.trainable_vars = None

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

        self.global_iter_counter = tf.Variable(0, name='global_iter_counter', trainable=False)
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, (None,) + self.input_shape, name='x')
            self.y_gt = tf.placeholder(tf.int32, shape=(None,), name='y_gt')
            self.is_phase_train = tf.placeholder(tf.bool, shape=tuple(), name='is_phase_train')

        self.__create_architecture(net_data, use_batch_norm)

        self.graph = tf.get_default_graph()
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        # please do not use the totality of the GPU memory.
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.sess = tf.Session(config=config)

    def __create_architecture(self, net_data, use_batch_norm):
        print 'CaffeNet::__create_architecture()'
        tr_vars = dict()
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

                # maxpool1
                # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
                kernel_height = 3
                kernel_width = 3
                s_h = 2
                s_w = 2
                padding = 'VALID'
                maxpool1 = tf.nn.max_pool(conv1, ksize=[1, kernel_height, kernel_width, 1],
                                          strides=[1, s_h, s_w, 1], padding=padding,
                                          name='maxpool')
                # lrn1
                # lrn(2, 2e-05, 0.75, name='norm1')
                radius = 2
                alpha = 2e-05
                beta = 0.75
                bias = 1.0
                maxpool1 = tf.nn.local_response_normalization(maxpool1,
                                                              depth_radius=radius,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias, name='lrn')

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

                # maxpool2
                # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
                kernel_height = 3
                kernel_width = 3
                s_h = 2
                s_w = 2
                padding = 'VALID'
                maxpool2 = tf.nn.max_pool(conv2, ksize=[1, kernel_height, kernel_width, 1],
                                          strides=[1, s_h, s_w, 1], padding=padding)

                # lrn2
                # lrn(2, 2e-05, 0.75, name='norm2')
                radius = 2
                alpha = 2e-05
                beta = 0.75
                bias = 1.0
                maxpool2 = tf.nn.local_response_normalization(maxpool2, depth_radius=radius,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias)

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

                if use_batch_norm:
                    print 'Using batch_norm after FC6'
                    self.fc6_bn = tflayers.batch_norm(self.fc6, decay=0.999,
                                                      is_training=self.is_phase_train,
                                                      trainable=False)
                    out = self.fc6_bn
                else:
                    out = self.fc6

                self.fc6_relu = tf.nn.relu(out, name='relu')

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
                if use_batch_norm:
                    print 'Using batch_norm after FC7'
                    self.fc7_bn = tflayers.batch_norm(self.fc7, decay=0.999,
                                                      is_training=self.is_phase_train,
                                                      trainable=False)
                    out = self.fc7_bn
                else:
                    out = self.fc7

                self.fc7_relu = tf.nn.relu(out, name='relu')

                self.fc7_keep_prob = tf.placeholder_with_default(1.0, tuple(),
                                                                 name='keep_prob_pl')
                self.fc7_dropout = tf.nn.dropout(self.fc7_relu, self.fc7_keep_prob, name='dropout')

            # fc(num_classes, relu=False, name='fc8')
            with tf.variable_scope('fc8'):
                num_inputs = int(self.fc7_dropout.get_shape()[1])
                num_outputs = self.num_classes
                tr_vars['fc8w'], tr_vars['fc8b'] = \
                    self.get_fc_weights(layer_index, net_data, num_inputs, num_outputs)
                layer_index += 1
                self.fc8 = tf.add(tf.matmul(self.fc7_dropout, tr_vars['fc8w']), tr_vars['fc8b'],
                                  name='fc')
                assert self.fc8.get_shape()[1] == self.num_classes, \
                    '{} != {}'.format(self.fc8.get_shape()[1], self.num_classes)

            self.logits = self.fc8
            with tf.variable_scope('output'):
                self.prob = tf.nn.softmax(self.fc8, name='prob')
        self.trainable_vars = tr_vars