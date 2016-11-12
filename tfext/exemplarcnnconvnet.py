################################################################################
# All conv net
# Copyright (c) 2016 Artsiom Sanakoyeu
################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import os


class ExemplarCnnConvNet(object):
    """
    Convnet
    WARNING! You should feed images in HxWxC BGR format!
    """

    class RandomInitType:
        GAUSSIAN = 0,
        XAVIER_UNIFORM = 1,
        XAVIER_GAUSSIAN = 2

    def __init__(self,
                 im_shape=(96, 96, 3),
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
        self.input_shape = im_shape
        self.device_id = device_id
        self.random_init_type = random_init_type
        self.batch_norm_decay = 0.99

        if len(self.input_shape) == 2:
            self.input_shape += (3,)
        assert len(self.input_shape) == 3

        self.global_iter_counter = tf.Variable(0, name='global_iter_counter', trainable=False)
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, (None,) + self.input_shape, name='x')
            self.y_gt = tf.placeholder(tf.int32, shape=(None,), name='y_gt')
            self.is_phase_train = tf.placeholder(tf.bool, shape=tuple(), name='is_phase_train')
            self.dropout_keep_prob = tf.placeholder_with_default(1.0, tuple(),
                                                                 name='dropout_keep_prob')


        with tf.device(self.device_id):

            self.conv1 = self.conv_relu(self.x, kernel_size=5,
                                       kernels_num=64, stride=1,
                                       name='conv1', batch_norm=False)
            self.maxpool1 = tf.nn.max_pool(self.conv1,
                                      ksize=[1, 48, 48, 1],
                                      strides=[1, 48, 48, 1],
                                      padding='SAME',
                                      name='maxpool1')
            self.pool1 = tf.nn.max_pool(self.conv1,
                                           ksize=[1, 3, 3, 1],
                                           strides=[1, 2, 2, 1],
                                           padding='SAME',
                                           name='maxpool1')


            self.conv2 = self.conv_relu(self.pool1, kernel_size=5,
                                   kernels_num=128, stride=1,
                                   name='conv2', batch_norm=False)
            self.maxpool2 = tf.nn.max_pool(self.conv2,
                                      ksize=[1, 24, 24, 1],
                                      strides=[1, 24, 24, 1],
                                      padding='SAME',
                                      name='maxpool2')

            self.pool2 = tf.nn.max_pool(self.conv2,
                                           ksize=[1, 3, 3, 1],
                                           strides=[1, 2, 2, 1],
                                           padding='SAME',
                                           name='maxpool2')

            self.conv3 = self.conv_relu(self.pool2, kernel_size=5,
                                   kernels_num=256, stride=1,
                                   name='conv3', batch_norm=False)

            self.maxpool3 = tf.nn.max_pool(self.conv3,
                                           ksize=[1, 12, 12, 1],
                                           strides=[1, 12, 12, 1],
                                           padding='SAME',
                                           name='maxpool3')

            self.conv4 = self.conv_relu(self.conv3, kernel_size=8,
                                   kernels_num=512, stride=1,
                                   name='conv4', batch_norm=False)

            self.maxpool4 = tf.nn.max_pool(self.conv4,
                                           ksize=[1, 9, 9, 1],
                                           strides=[1, 9, 9, 1],
                                           padding='VALID',
                                           name='maxpool4')
        self.graph = tf.get_default_graph()
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        # please do not use the totality of the GPU memory.
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.sess = tf.Session(config=config)



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
        if num_layers > 4:
            raise ValueError('You can restore only 4 layers')
        if num_layers == 0:
            return
        # if not restore_iter_counter:
        #     raise ValueError('We can restore only everything including iter_counter')

        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, snapshot_path)

    def conv_relu(self, input_tensor, kernel_size, kernels_num, stride, batch_norm=True,
                      group=1, name=None):

        with tf.variable_scope(name) as scope:
            assert int(input_tensor.get_shape()[3]) % group == 0
            num_input_channels = int(input_tensor.get_shape()[3]) / group
            w, b = self.get_conv_weights(kernel_size, num_input_channels, kernels_num)
            conv = self.conv(input_tensor, w, b, stride, padding="SAME", group=group)
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

    @staticmethod
    def conv(input_tensor, kernel, biases, stride, padding="VALID", group=1):

        c_i = input_tensor.get_shape()[-1]
        assert c_i % group == 0
        assert kernel.get_shape()[3] % group == 0

        def convolve(inp, w, name=None):
            return tf.nn.conv2d(inp, w, [1, stride, stride, 1], padding=padding, name=name)

        if group == 1:
            conv = convolve(input_tensor, kernel)
        else:
            input_groups = tf.split(3, group, input_tensor)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        # TODO: no need to reshape?
        return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:],
                          name='conv')

    def get_conv_weights(self, kernel_size, num_input_channels, kernels_num,
                             weight_std=0.01, bias_init_value=0.1):

        w = self.random_weight_variable((kernel_size, kernel_size,
                                         num_input_channels,
                                         kernels_num),
                                        stddev=weight_std)
        b = self.random_bias_variable((kernels_num,), value=bias_init_value)
        return w, b

    def random_weight_variable(self, shape, stddev=0.01):
        """
        stddev is used only for RandomInitType.GAUSSIAN
        """
        if self.random_init_type == self.RandomInitType.GAUSSIAN:
            initial = tf.truncated_normal(shape, stddev=stddev)
            return tf.Variable(initial, name='weight')
        elif self.random_init_type == self.RandomInitType.XAVIER_GAUSSIAN:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=False))
        elif self.random_init_type == self.RandomInitType.XAVIER_UNIFORM:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=True))
        else:
            raise ValueError('Unknown random_init_type')


    @staticmethod
    def random_bias_variable(shape, value=0.1):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial, name='bias')


if __name__ == "__main__":

    path_to_snapshot = '/export/home/mbautist/Desktop/workspace/cnn_similarities/tjprj/data/exemplar_cnn/'
    net = ExemplarCnnNet(device_id=0,
                         gpu_memory_fraction=0.4, random_init_type=ExemplarCnnNet.RandomInitType.XAVIER_GAUSSIAN)
    a = 0
