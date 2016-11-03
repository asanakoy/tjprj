################################################################################
# Deep model for STL-10 training
# Copyright (c) 2016 Artsiom Sanakoyeu
################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers


class Stlnet(object):
    """
    STLNet
    WARNING! You should feed images in HxWxC BGR format!
    """

    class RandomInitType:
        GAUSSIAN = 0,
        XAVIER_UNIFORM = 1,
        XAVIER_GAUSSIAN = 2

    def __init__(self,
                 im_shape=(96, 96, 3),
                 num_classes=1,
                 device_id='/gpu:0',
                 random_init_type=RandomInitType.XAVIER_GAUSSIAN,
                 gpu_memory_fraction=None):
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
            conv1 = self.conv_relu(self.x, kernel_size=5,
                                   kernels_num=64, stride=2,
                                   name='conv1')
            conv2 = self.conv_relu(conv1, kernel_size=1,
                                   kernels_num=160, stride=1,
                                   name='conv2')
            conv3 = self.conv_relu(conv2, kernel_size=1,
                                   kernels_num=96, stride=1,
                                   name='conv3')

            maxpool3 = tf.nn.max_pool(conv3,
                                      ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID',
                                      name='maxpool3')

            conv4 = self.conv_relu(maxpool3, kernel_size=5,
                                   kernels_num=192, stride=2,
                                   name='conv4')
            conv5 = self.conv_relu(conv4, kernel_size=1,
                                   kernels_num=192, stride=1,
                                   name='conv5')
            conv6 = self.conv_relu(conv5, kernel_size=1,
                                   kernels_num=192, stride=1,
                                   name='conv6')

            maxpool6 = tf.nn.max_pool(conv6,
                                      ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID',
                                      name='maxpool6')

            conv7 = self.conv_relu(maxpool6, kernel_size=3,
                                   kernels_num=192, stride=1,
                                   name='conv7')
            conv8 = self.conv_relu(conv7, kernel_size=1,
                                   kernels_num=192, stride=1,
                                   name='conv8')
            conv9 = self.conv_relu(conv8, kernel_size=1,
                                   kernels_num=192, stride=1,
                                   name='conv9')

            conv10 = self.conv_relu(conv9, kernel_size=3,
                                    kernels_num=256, stride=1,
                                    batch_norm=False,
                                    name='conv10')

            maxpool10 = tf.nn.max_pool(conv10,
                                       ksize=[1, 3, 3, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='VALID',
                                       name='maxpool10')
            dropout10 = tf.nn.dropout(maxpool10, self.dropout_keep_prob, name='dropout10')

            conv11 = self.conv_relu(dropout10, kernel_size=3,
                                    kernels_num=128, stride=1,
                                    batch_norm=False,
                                    name='conv11')
            dropout11 = tf.nn.dropout(conv11, self.dropout_keep_prob, name='dropout11')

            self.fc12 = self.fc_relu(dropout11,
                                     num_outputs=num_classes,
                                     relu=False,
                                     weight_std=0.01, bias_init_value=0.0,
                                     name='fc12')[0]

            self.fc_stl10 = self.fc_relu(dropout11,
                                     num_outputs=10,
                                     relu=False,
                                     weight_std=0.01, bias_init_value=0.0,
                                     name='fc_stl10')[0]

            with tf.variable_scope('output'):
                self.prob = tf.nn.softmax(self.fc12, name='prob')
                self.prob_stl10 = tf.nn.softmax(self.fc_stl10, name='prob_stl10')

            fc12_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc12/weight:0")[0]
            fc12_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc12/bias:0")[0]
            self.reset_fc12_op = tf.initialize_variables([fc12_w, fc12_b], name='reset_fc12')

            fc12_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc_stl10/weight:0")[0]
            fc12_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc_stl10/bias:0")[0]
            self.reset_fc_stl10_op = tf.initialize_variables([fc12_w, fc12_b], name='reset_fc_stl10')

        self.graph = tf.get_default_graph()
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        # please do not use the totality of the GPU memory.
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.sess = tf.Session(config=config)

    def restore_from_snapshot(self, snapshot_path, num_layers):
        """
        :param snapshot_path: path to the snapshot file
        :param num_layers: number layers to restore from the snapshot
                            (conv1 is the #1, fc8 is the #8)
        :param restore_iter_counter: if True restore global_iter_counter from the snapshot

        WARNING! A call of sess.run(tf.initialize_all_variables()) after restoring from snapshot
                 will overwrite all variables and set them to initial state.
                 Call restore_from_snapshot() only after sess.run(tf.initialize_all_variables())!
        """
        if num_layers > 12 or num_layers < 11:
            raise ValueError('You can restore only 11 or 12 layers.')
        if num_layers == 0:
            return
        saver = tf.train.Saver()
        saver.restore(self.sess, snapshot_path)
        if num_layers == 11:
            self.reset_fc12()
            self.reset_fc_stl10()

    def reset_fc12(self):
        print 'Resetting fc12 to random'
        self.sess.run(self.reset_fc12_op)

    def reset_fc_stl10(self):
        print 'Resetting fc_stl10 to random'
        self.sess.run(self.reset_fc_stl10_op)

    def get_conv_weights(self, kernel_size, num_input_channels, kernels_num,
                         weight_std=0.01, bias_init_value=0.1):
        w = self.random_weight_variable((kernel_size, kernel_size,
                                         num_input_channels,
                                         kernels_num),
                                        stddev=weight_std)
        b = self.random_bias_variable((kernels_num,), value=bias_init_value)
        return w, b

    def get_fc_weights(self, num_inputs, num_outputs, weight_std=0.005, bias_init_value=0.1):
        w = self.random_weight_variable((num_inputs, num_outputs), stddev=weight_std)
        b = self.random_bias_variable((num_outputs,), value=bias_init_value)
        return w, b

    def random_weight_variable(self, shape, stddev=0.01):
        """
        stddev is used only for RandomInitType.GAUSSIAN
        """
        if self.random_init_type == Stlnet.RandomInitType.GAUSSIAN:
            initial = tf.truncated_normal(shape, stddev=stddev)
            return tf.Variable(initial, name='weight')
        elif self.random_init_type == Stlnet.RandomInitType.XAVIER_GAUSSIAN:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=False))
        elif self.random_init_type == Stlnet.RandomInitType.XAVIER_UNIFORM:
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
    def conv(input_tensor, kernel, biases, stride, padding="VALID"):
        conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding=padding)
        # TODO: no need to reshape?
        return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:],
                          name='conv')

    def conv_relu(self, input_tensor, kernel_size, kernels_num, stride, batch_norm=True,
                  name=None):
        with tf.variable_scope(name) as scope:
            num_input_channels = int(input_tensor.get_shape()[3])
            w, b = self.get_conv_weights(kernel_size, num_input_channels, kernels_num)
            conv = Stlnet.conv(input_tensor, w, b, stride, padding="SAME")
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
            w, b = self.get_fc_weights(num_inputs, num_outputs,
                                       weight_std=weight_std,
                                       bias_init_value=bias_init_value)
            fc_relu = None

            input_tensor_reshaped = tf.reshape(input_tensor, [-1, num_inputs])
            if relu:
                fc = tf.add(tf.matmul(input_tensor_reshaped, w), b, name='fc')
                fc_relu = tf.nn.relu(fc, name=name)
            else:
                fc = tf.add(tf.matmul(input_tensor_reshaped, w), b, name=name)
        return fc, fc_relu
