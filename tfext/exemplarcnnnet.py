################################################################################
# All conv net
# Copyright (c) 2016 Artsiom Sanakoyeu
################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import os


class ExemplarCnnNet(object):
    """
    Convnet
    WARNING! You should feed images in HxWxC BGR format!
    """

    class RandomInitType:
        GAUSSIAN = 0,
        XAVIER_UNIFORM = 1,
        XAVIER_GAUSSIAN = 2

    def __init__(self,
                 init_model=None,
                 im_shape=(32, 32, 3),
                 num_classes=1,
                 num_layers_to_init=0,
                 device_id='/gpu:0',
                 random_init_type=RandomInitType.XAVIER_GAUSSIAN,
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
        self.num_layers_to_init = num_layers_to_init
        self.input_shape = im_shape
        self.device_id = device_id
        self.random_init_type = random_init_type
        self.batch_norm_decay = 0.99

        if len(self.input_shape) == 2:
            self.input_shape += (3,)
        assert len(self.input_shape) == 3

        if self.num_layers_to_init > 5 or self.num_layers_to_init < 0:
            raise ValueError('Number of layer to init must be in [0, 5] ({} provided)'.
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
            self.dropout_keep_prob = tf.placeholder_with_default(1.0, tuple(),
                                                                 name='dropout_keep_prob')

        self.__create_architecture(net_data, use_batch_norm=False)

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
        print 'ExemplarCnnNet::__create_architecture()'
        tr_vars = dict()
        with tf.device(self.device_id):
            layer_index = 0

            # conv(5, 5, 64, 1, 1, padding='VALID', name='conv1')
            with tf.variable_scope('conv1'):
                kernel_height = 5
                kernel_width = 5
                kernels_num = 64
                num_input_channels = int(self.x.get_shape()[3])
                s_h = 1  # stride by the H dimension (height)
                s_w = 1  # stride by the W dimension (width)
                tr_vars['conv1w'], tr_vars['conv1b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels, kernels_num)
                layer_index += 1
                conv1_in = ExemplarCnnNet.conv(self.x, tr_vars['conv1w'], tr_vars['conv1b'],
                                        kernel_height, kernel_width,
                                        kernels_num, s_h, s_w, padding="SAME")
                self.conv1 = tf.nn.relu(conv1_in)

                # maxpool1
                # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
                kernel_height = 3
                kernel_width = 3
                s_h = 2
                s_w = 2
                padding = 'SAME'
                maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, kernel_height, kernel_width, 1],
                                          strides=[1, s_h, s_w, 1], padding=padding,
                                          name='pool1')

            # conv(5, 5, 128, 1, 1,, name='conv2')
            with tf.variable_scope('conv2'):
                kernel_height = 5
                kernel_width = 5
                kernels_num = 128
                num_input_channels = int(maxpool1.get_shape()[3])
                s_h = 1
                s_w = 1
                tr_vars['conv2w'], tr_vars['conv2b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels, kernels_num)
                layer_index += 1
                conv2_in = ExemplarCnnNet.conv(maxpool1, tr_vars['conv2w'], tr_vars['conv2b'],
                                        kernel_height, kernel_width,
                                        kernels_num, s_h, s_w, padding="SAME")
                self.conv2 = tf.nn.relu(conv2_in, name='relu')

                # maxpool2
                # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
                kernel_height = 3
                kernel_width = 3
                s_h = 2
                s_w = 2
                padding = 'SAME'
                maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, kernel_height, kernel_width, 1],
                                          strides=[1, s_h, s_w, 1], padding=padding)

            # conv(5, 5, 256, 1, 1, name='conv3')
            with tf.variable_scope('conv3'):
                kernel_height = 5
                kernel_width = 5
                kernels_num = 256
                num_input_channels = int(maxpool2.get_shape()[3])
                s_h = 1
                s_w = 1
                tr_vars['conv3w'], tr_vars['conv3b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels, kernels_num)
                layer_index += 1
                conv3_in = ExemplarCnnNet.conv(maxpool2, tr_vars['conv3w'], tr_vars['conv3b'],
                                        kernel_height, kernel_width,
                                        kernels_num, s_h, s_w, padding="SAME")

                self.conv3 = tf.nn.relu(conv3_in, 'relu')


            # # fc(512, name='fc6')
            # with tf.variable_scope('fc6'):
            #     num_inputs = int(np.prod(self.conv3.get_shape()[1:]))
            #     num_outputs = 512
            #     tr_vars['fc6w'], tr_vars['fc6b'] = \
            #         self.get_fc_weights(layer_index, net_data, num_inputs, num_outputs)
            #     layer_index += 1
            #     self.fc6 = tf.add(tf.matmul(
            #         tf.reshape(self.conv3,
            #                    [-1, int(np.prod(self.conv3.get_shape()[1:]))]
            #                    ),
            #         tr_vars['fc6w']),
            #         tr_vars['fc6b'], name='fc')
            #
            #     if use_batch_norm:
            #         print 'Using batch_norm after FC6'
            #         self.fc6_bn = tflayers.batch_norm(self.fc6, decay=0.999,
            #                                           is_training=self.is_phase_train,
            #                                           trainable=False)
            #         out = self.fc6_bn
            #     else:
            #         out = self.fc6
            #
            #     self.fc6_relu = tf.nn.relu(out, name='relu')
            #
            #     self.fc6_keep_prob = tf.placeholder_with_default(1.0, tuple(),
            #                                                      name='keep_prob_pl')
            #     self.fc6_dropout = tf.nn.dropout(self.fc6_relu, self.fc6_keep_prob, name='dropout')


            # # conv(512, name='conv4')
            # with tf.variable_scope('conv4'):
            #     kernels_num = 512
            #     num_inputs = int(np.prod(self.conv3.get_shape()[1:]))
            #     with tf.variable_scope('tmp'):
            #         fc_w, fc_b = self.get_fc_weights(layer_index, net_data, num_inputs, kernels_num)
            #     layer_index += 1
            #     self.conv4, tr_vars['conv4w'], tr_vars['conv4b'] = self.fc_to_convolution(self.conv3, fc_w, fc_b,
            #                                                                                kernels_num, use_relu=True)

        self.trainable_vars = tr_vars


    def restore_from_snapshot(self, snapshot_path, num_layers, restore_iter_counter=True):
        """
        :param snapshot_path: path to the snapshot file
        :param num_layers: number layers to restore from the snapshot
                            (conv1 is the #1, fc8 is the #8)
        :param restore_iter_counter: if True restore global_iter_counter from the snapshot

        WARNING! A call of sess.run(tf.initialize_all_variables()) after restoring from snapshot
                 will overwrite all variables and set them to initial state.
                 Call restore_from_snapshot() only after sess.run(tf.initialize_all_variables())!
        """
        if num_layers > 5 or num_layers < 3:
            raise ValueError('You can restore only 4 or 5 layers.')
        if num_layers == 0:
            return
        if not restore_iter_counter:
            raise ValueError('We can restore only everything including iter_counter')

        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, snapshot_path)
            # _, w, b = self.fc_to_convolution(self.conv3, weights=, biases=, out_channel=512, use_relu=True)

    def get_conv_weights(self, layer_index, net_data, kernel_height, kernel_width,
                         num_input_channels, kernels_num):

        layer_names = ['conv{}'.format(i) for i in xrange(1, 6)] + \
                      ['fc{}'.format(i) for i in xrange(6, 9)]
        wights_std = [0.01] * 5 + [0.005, 0.005, 0.01]
        bias_init_values = [0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0]

        l_name = layer_names[layer_index]
        if net_data is not None and layer_index < self.num_layers_to_init:
            assert net_data[l_name]['weights'].shape == (kernel_height, kernel_width,
                                                         num_input_channels,
                                                         kernels_num)
            assert net_data[l_name]['biases'].shape == (kernels_num,)

        if layer_index >= self.num_layers_to_init or net_data is None:
            print 'Initializing {} with random'.format(l_name)
            w = self.random_weight_variable((kernel_height, kernel_width,
                                             num_input_channels,
                                             kernels_num),
                                            stddev=wights_std[layer_index])
            b = self.random_bias_variable((kernels_num,), value=bias_init_values[layer_index])
        else:
            w = tf.Variable(net_data[l_name]['weights'], name='weight')
            b = tf.Variable(net_data[l_name]['biases'], name='bias')
        return w, b

    def get_fc_weights(self, layer_index, net_data, num_inputs, num_outputs,
                           weight_std=None,
                           bias_init_value=None):

            if layer_index <= 4:
                if weight_std is not None or bias_init_value is not None:
                    raise ValueError('std and bias must be None for layers 1..4, they are set up automatically')
                layer_names = ['conv{}'.format(i) for i in xrange(1, 4)] + \
                              ['fc{}'.format(i) for i in xrange(6, 7)]
                weights_stds = [0.01] * 5 + [0.005, 0.005, 0.01]
                bias_init_values = [0.0, 0.1, 0.0, 0.1]
                l_name = layer_names[layer_index]

                weight_std = weights_stds[layer_index]
                bias_init_value = bias_init_values[layer_index]
            else:
                l_name = 'layer {}'.format(layer_index)
                if weight_std is None or bias_init_value is None:
                    raise ValueError('std and bias must be provided for all layers beyond 1..8')

            if net_data is not None and layer_index < self.num_layers_to_init:
                assert net_data[l_name]['weights'].shape == (num_inputs, num_outputs)
                assert net_data[l_name]['biases'].shape == (num_outputs,)

            if layer_index >= self.num_layers_to_init or net_data is None:
                print 'Initializing {} with random'.format(l_name)
                w = self.random_weight_variable((num_inputs, num_outputs),
                                                stddev=weight_std)
                b = self.random_bias_variable((num_outputs,), value=bias_init_value)
            else:
                w = tf.Variable(net_data[l_name]['weights'], name='weight')
                b = tf.Variable(net_data[l_name]['biases'], name='bias')
            return w, b

    def fc_to_convolution(self, in_put, weights, biases, out_channel, use_relu=True):

        """
        :param in_put:
        :param out_channel:
        :param layer_name:
        :param use_relu:
        :return:
        """
        input_shape = in_put.get_shape()
        assert len(input_shape) == 4
        height, width, in_channel = input_shape[1:]
        reshape_weights = tf.reshape(weights,
                                     shape=[tf.to_int32(height), tf.to_int32(width),
                                            tf.to_int32(in_channel), out_channel])
        convolution_output = tf.nn.conv2d(input=in_put, filter=reshape_weights, strides=[1, 1, 1, 1],
                                          padding="VALID")

        output = tf.nn.bias_add(convolution_output, biases)
        if use_relu:
            output = tf.nn.relu(output)
        w = tf.Variable(reshape_weights, name='weight')
        b = tf.Variable(biases, name='bias')
        return output, w, b

    def random_weight_variable(self, shape, stddev=0.01):
        """
        stddev is used only for RandomInitType.GAUSSIAN
        """
        if self.random_init_type == ExemplarCnnNet.RandomInitType.GAUSSIAN:
            initial = tf.truncated_normal(shape, stddev=stddev)
            return tf.Variable(initial, name='weight')
        elif self.random_init_type == ExemplarCnnNet.RandomInitType.XAVIER_GAUSSIAN:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=False))
        elif self.random_init_type == ExemplarCnnNet.RandomInitType.XAVIER_UNIFORM:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=True))
        else:
            raise ValueError('Unknown random_init_type')

    @staticmethod
    def random_bias_variable(shape, value=0.1):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial, name='bias')

    @staticmethod
    def conv(input, kernel, biases, kernel_height, kernel_width, kernels_num, s_h, s_w, padding="VALID", group=1):
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
            input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
            kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(axis=3, values=output_groups)
        return tf.reshape(tf.nn.bias_add(conv, biases),
                          [-1] + conv.get_shape().as_list()[1:], name='conv')

    def conv_relu(self, input_tensor, kernel_size, kernels_num, stride, batch_norm=True,
                  group=1, name=None):
        with tf.variable_scope(name) as scope:
            assert int(input_tensor.get_shape()[3]) % group == 0
            num_input_channels = int(input_tensor.get_shape()[3]) / group
            w, b = self.get_conv_weights(kernel_size, num_input_channels, kernels_num)
            conv = ExemplarCnnNet.conv(input_tensor, w, b, stride, padding="SAME", group=group)
            if batch_norm:
                conv = tf.cond(self.is_phase_train,
                               lambda: tflayers.batch_norm(conv,
                                                           decay=self.batch_norm_decay,
                                                           is_training=True,
                                                           trainable=True,
                                                           reuse=None,
                                                           scope=scope),
                               lambda: tflayers.batch_norm(conv,
                                                           decay=self.batch_norm_decay,
                                                           is_training=False,
                                                           trainable=True,
                                                           reuse=True,
                                                           scope=scope))
            conv = tf.nn.relu(conv, name=name)
        return conv

    def fc_relu(self, input_tensor, num_outputs, relu=False, weight_std=0.005,
                bias_init_value=0.1, name=None):
        with tf.variable_scope(name):
            num_inputs = int(np.prod(input_tensor.get_shape()[1:]))
            w, b = self.get_fc_weights(layer_index=5, net_data=None, num_inputs=num_inputs, num_outputs=num_outputs,
                                       weight_std=weight_std, bias_init_value=bias_init_value)

            fc_relu = None
            input_tensor_reshaped = tf.reshape(input_tensor, [-1, num_inputs])
            if relu:
                fc = tf.add(tf.matmul(input_tensor_reshaped, w), b, name='fc')
                fc_relu = tf.nn.relu(fc, name=name)
            else:
                fc = tf.add(tf.matmul(input_tensor_reshaped, w), b, name=name)
        return fc, fc_relu

if __name__ == "__main__":

    path_to_snapshot = '/export/home/mbautist/Desktop/workspace/cnn_similarities/tjprj/data/exemplar_cnn/weights.npy'
    with tf.Graph().as_default():
        net = ExemplarCnnNet(init_model=path_to_snapshot, num_layers_to_init=4, num_classes=1000, device_id=0,
                             gpu_memory_fraction=0.4, random_init_type=ExemplarCnnNet.RandomInitType.XAVIER_GAUSSIAN)
        net.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(net.sess, '/export/home/mbautist/Desktop/exemplar_cnn_conv')
        saver.restore(net.sess, '/export/home/mbautist/Desktop/exemplar_cnn_conv')
        a = 0