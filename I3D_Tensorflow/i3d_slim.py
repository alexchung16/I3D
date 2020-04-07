#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : i3d_slim.py
# @ Description: tensorflow==1.14.0 tensorflow_probability==0.7.0 pip install dm-sonnet==1.32
#                https://github.com/tensorflow/agents/issues/91
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/2/24 上午9:40
# @ Software   : PyCharm
#-------------------------------------------------------
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

import I3D_Tensorflow.i3d as i3d

class I3D():
    """
      I3D model
    """

    def __init__(self, num_classes, learning_rate=0.01, decay_step_first=10000, decay_step_second=20000,
                 decay_step_third=30000, decay_step_fourth=400000, learning_rate_first=0.0005,
                 learning_rate_second=0.0003, learning_rate_third=0.00002, learning_rate_fourth=0.00001,
                 momentum_rate=0.9, keep_prob=0.8):

        self.num_classes = num_classes

        self.rgb_input_data = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 3],
                                                       name="rgb_video")
        # convert size scale to (-1, 1)
        # self.rgb_input_data = self.image_rescale(self.rgb_input_data)

        self.flow_input_data = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 2],
                                                       name="flow_video")

        self.input_label = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_classes],
                                                    name="label")
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=(), name="is_training")

        self.learning_rate_value = [learning_rate, learning_rate_first, learning_rate_second, learning_rate_third,
                                    learning_rate_fourth]
        self.learning_rate_boundary = [decay_step_first, decay_step_second, decay_step_third, decay_step_fourth]

        self.momentum_rate = momentum_rate

        self.keep_prob = keep_prob

        # # is_training flag
        self.global_step = tf.train.get_or_create_global_step()
        self.rgb_logits, self.flow_logits = self.inference()
        self.model_logits = self.rgb_logits + self.flow_logits

        self.raw_model_variables = self.get_model_variables()

        self.rgb_loss, self.flow_loss = self.losses(self.rgb_logits, self.flow_logits, self.input_label)
        self.train = self.training(self.rgb_loss, self.flow_loss)
        self.accuracy = self.get_accuracy()


    def inference(self):
        """
        model inference
        :return:
        """
        rgb_logits, flow_logits = self.i3d_net(rgb_input=self.rgb_input_data,
                                               flow_input=self.flow_input_data,
                                               num_classes=self.num_classes,
                                               keep_prob=self.keep_prob,
                                               is_training=self.is_training)
        return rgb_logits, flow_logits

    def i3d_net(self, rgb_input, flow_input, num_classes, keep_prob, is_training):
        """
        construct i3d net
        :return:
        """

        with tf.variable_scope('RGB'):
            # insert i3d model
            model = InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')

            rgb_logits = model(rgb_input, is_training=is_training, keep_prob=keep_prob)

            rgb_logits = tf.nn.dropout(rgb_logits, keep_prob)
            # To change 400 classes to custom classes
            rgb_logits = tf.layers.dense(rgb_logits, num_classes, use_bias=True, name='Dense_Logits')


        with tf.variable_scope('Flow'):
            # insert i3d model
            model = InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')

            flow_logits = model(flow_input, is_training=is_training, keep_prob=keep_prob)

            flow_logits = tf.nn.dropout(flow_logits, keep_prob)
            # To change 400 classes to custom classes
            flow_logits = tf.layers.dense(flow_logits, num_classes, use_bias=True, name='Dense_Logits')

        return rgb_logits, flow_logits


    def get_model_variables(self):
        """
        get model_variable
        :return:
        """

        return tf.global_variables()



    def losses(self, rgb_logits, flow_logits, labels, weight_lambda=7e-7):
        """
        :return:
        """
        rgb_loss_op = self.get_rgb_flow_loss('RGB', labels=labels, logits=rgb_logits,
                                             weight_lambda=weight_lambda)
        flow_loss_op = self.get_rgb_flow_loss('Flow', labels=labels, logits=flow_logits,
                                              weight_lambda=weight_lambda)

        return rgb_loss_op, flow_loss_op


    def training(self, rgb_loss, flow_loss):
        """

        :param loss:
        :param global_step:
        :return:
        """

        learning_rate = tf.train.piecewise_constant(self.global_step,
                                                    boundaries=self.learning_rate_boundary,
                                                    values= self.learning_rate_value)

        rgb_variable = tf.global_variables(scope='RGB')
        flow_variable = tf.global_variables(scope='Flow')
        #
        rgb_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(rgb_update_ops):
            rgb_train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                      momentum=self.momentum_rate).minimize(rgb_loss,
                                                                                            global_step=self.global_step,
                                                                                            var_list=rgb_variable)
        flow_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Flow')
        with tf.control_dependencies(flow_update_ops):
            flow_train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                       momentum=self.momentum_rate).minimize(flow_loss,
                                                                                             global_step=self.global_step,
                                                                                             var_list=flow_variable)
        train_op = tf.group(rgb_train_op, flow_train_op)

        return train_op


    def get_accuracy(self):
        """

        :return:
        """
        rgb_acc = tf.equal(tf.argmax(self.rgb_logits, 1), tf.argmax(self.input_label, 1))
        flow_acc = tf.equal(tf.argmax(self.flow_logits, 1), tf.argmax(self.input_label, 1))
        model_acc = tf.equal(tf.argmax(self.model_logits, 1), tf.argmax(self.input_label, 1))

        # accuracy mean
        # rgb_accuracy = tf.reduce_mean(tf.cast(rgb_acc, tf.float32))
        # flow_accuracy = tf.reduce_mean(tf.cast(flow_acc, tf.float32))
        # model_accuracy = tf.reduce_mean(tf.cast(model_acc, tf.float32))


        return rgb_acc, flow_acc, model_acc


    def predict(self):

        predict = tf.nn.softmax(self.model_logits)

        return tf.argmax(predict)


    def image_rescale(self, image):
        """
        convert image pixel size number to [-1, 1]
        :param image:
        :return:
        """
        # [0, 255]=>[0, 1]
        image = tf.divide(tf.cast(image, dtype=tf.float32), 255.)
        # [0, 1] => [-0.5, 0.5]
        image = tf.subtract(image, 0.5)
        # [-0.5, 0.5] => [-1.0, 1.0]
        image = tf.multiply(image, 2.0)

        return image

    def fill_feed_dict(self, rgb_video_feed, flow_video_feed, label_feed, is_training=False):
        """

        :param rgb_video_feed:
        :param flow_video_feed:
        :param label_feed:
        :param is_training:
        :return:
        """
        feed_dict = {
            self.rgb_input_data: rgb_video_feed,
            self.flow_input_data: flow_video_feed,
            self.input_label: label_feed,
            self.is_training: is_training
        }

        return feed_dict

    def load_pretrain_model(self, sess, rgb_model_path, flow_model_path, load_Logits=False):
        """
        restore i3d rgb and flow pretrain model
        :param rgb_model_path:
        :param flow_model_path:
        :param load_logits:
        :return:
        """

        flags = ['RGB', 'Flow']
        model_paths = [rgb_model_path, flow_model_path]

        for flag, model_path in zip(flags, model_paths):
            try:
                self.restore_rgb_flow_mdoel(sess, flag, model_path, load_Logits=load_Logits)
                print('Successful restore I3D {0} mdoel'.format(flag))

            except Exception as e:
                print('Failed restore I3D {0} model from {1}, for\n \t{2}'.format(flag, model_path, e))
                continue

    def get_rgb_flow_loss(self, mode='RGB', labels=None, logits=None, weight_lambda=7e-7):
        """
        get rgb or flow loss
        :param mode:
        :param labels:
        :param logits:
        :param weight_lambda:
        :return:
        """
        # add L2 regularization to weights
        for variable in tf.global_variables(scope=mode):
            var_split = variable.name.split('/')
            if var_split[-1] == 'w:0' or var_split[-1] == 'kernel:0':
                weight_l2 = tf.nn.l2_loss(variable)
                tf.add_to_collection('weight_l2', weight_l2)
        loss_weight = tf.add_n(tf.get_collection('weight_l2'), 'loss_weight')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                      logits=logits))
        total_loss = loss + weight_lambda * loss_weight
        tf.summary.scalar('{0} loss'.format(mode), loss)
        tf.summary.scalar('{0} loss_weight'.format(mode), loss_weight)
        tf.summary.scalar('{0} total_loss'.format(mode), total_loss)


        return total_loss


    def restore_rgb_flow_mdoel(self, sess, model_flag, model_path, load_Logits=None):
        """
        restore i3d rgb and flow pretrain model
        :param sess:
        :param model_flag: 'RGB' | 'FLow
        :param model_path:
        :param load_Logits:
        :return:
        """

        # load RGB model
        src_scope = '{0}/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3'.format(model_flag)
        dst_scope = '{0}/inception_i3d/Mixed_5b/Branch_2/Conv3d_0b_3x3'.format(model_flag)

        dense_layer_scope = '{0}/Dense_Logits'.format(model_flag)

        self.load_rename_checkpoint_variable(checkpoint_path=model_path, src_scope=src_scope, dst_scope=dst_scope)


        # # generate rgb and flow restore op
        custom_scope = [dst_scope, dense_layer_scope]
        if load_Logits is None or load_Logits is False:
            custom_scope.append('{0}/inception_i3d/Logits'.format(model_flag))

        saver = self.get_saver(model_flag, custom_scope=custom_scope)
        saver.restore(sess, save_path=model_path)

        return True

    def get_saver(self, mode_scope, custom_scope):
        """
        remove custom variable
        :param variables:
        :param custom_scope:
        :return:
        """

        model_variable = tf.global_variables(mode_scope)
        for scope in custom_scope:
            custom_variable = tf.global_variables(scope=scope)

            [model_variable.remove(variable) for variable in custom_variable]

        # remove extend variables
        model_variable = [var for var in model_variable if var in self.raw_model_variables]

        saver = tf.train.Saver(var_list=model_variable, reshape=True)

        return saver

    def load_rename_checkpoint_variable(self, checkpoint_path, src_scope, dst_scope):
        """
        load variable from checkpoint model to graph where the aspect variable name is different
        :param checkpoint_path:
        :param src_scope:
        :param dst_scope:
        :return:
        """
        dst_variable = tf.global_variables(scope=dst_scope)

        # filter momentum variable or remove extend variables
        dst_variable = [var for var in dst_variable if var in self.raw_model_variables]


        dst_var_names = [var.op.name for var in dst_variable]

        src_var_names = [var_name.replace(dst_scope, src_scope) for var_name in dst_var_names]

        # src_name with dst_name
        var_name_map = {}
        # dst_name with dst_var
        dst_var_map = {}
        for src_var_name, dst_var_name, dst_var in zip(src_var_names, dst_var_names, dst_variable):
            var_name_map[src_var_name] = dst_var_name
            dst_var_map[dst_var_name] = dst_var
        # execute assign operation
        for var_name, var_shape in tf.train.list_variables(checkpoint_path):
            # get_variable
            var_value = tf.train.load_variable(checkpoint_path, var_name)

            if var_name in src_var_names:
                dst_var = dst_var_map[var_name_map[var_name]]
                # ensure the checkpoint variable shape same as  graph variable
                var_value = tf.reshape(var_value, dst_var.shape)
                # assign value to variable of graph
                tf.assign(dst_var, value=var_value)
            else:
                pass

        return True


class Unit3D(snt.AbstractModule):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        """Connects the module to inputs.

            Args:
              inputs: Inputs to the Unit3D component.
              is_training: whether to use training mode for snt.BatchNorm (boolean).

            Returns:
              Outputs from the module.
            """
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,
                         use_bias=self._use_bias)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net


class InceptionI3d(snt.AbstractModule):
    """Inception-v1 I3D architecture.

    The model is introduced in:

    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
    Joao Carreira, Andrew Zisserman
    https://arxiv.org/pdf/1705.07750v1.pdf.

    See also the Inception architecture, introduced in:

    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    VALID_ENDPOINTS = (
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d'):
        """Initializes I3D model instance.

        Args:
            num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
            spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
            final_endpoint: The model contains many possible endpoints.
            final_endpoint` specifies the last endpoint for the model to be built
            up to. In addition to the output at `final_endpoint`, all the outputs
            at endpoints up to `final_endpoint` will also be returned, in a
            dictionary. `final_endpoint` must be one of
            InceptionI3d.VALID_ENDPOINTS (default 'Logits').
            name: A string (optional). The name of this module.

        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        super(InceptionI3d, self).__init__(name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint


    def _build(self, inputs, keep_prob=1.0, is_training=False):
        """Connects the model to inputs.
            Args:
                inputs: Inputs to the model, which should have dimensions
                        `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
                is_training: whether to use training mode for snt.BatchNorm (boolean).
                keep_prob: Probability for the tf.nn.dropout layer (float in [0, 1)).
            Returns:
                  A tuple consisting of:
                    1. Network output at location `self._final_endpoint`
                Raises:
                  ValueError: if `self._final_endpoint` is not recognized.
                """

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        # Conv3d_1a_7x7
        net = Unit3D(output_channels=64,
                     kernel_shape=(7, 7, 7),
                     stride=(2, 2, 2),
                     name='Conv3d_1a_7x7')(net, is_training=is_training)

        # MaxPool3d_2a_3x3
        net = tf.nn.max_pool3d(input=net, ksize=(1, 1, 3, 3, 1), strides=(1, 1, 2, 2, 1), padding=snt.SAME,
                               name='MaxPool3d_2a_3x3')
        # Conv3d_2b_1x1
        net = Unit3D(output_channels=64,
                     kernel_shape=(1, 1, 1),
                     stride=(1, 1, 1),
                     name='Conv3d_2b_1x1')(net, is_training=is_training)
        # Conv3d_2c_3x3
        net = Unit3D(output_channels=192,
                     kernel_shape=(3, 3, 3),
                     stride=(1, 1, 1),
                     name='Conv3d_2c_3x3')(net, is_training=is_training)

        # MaxPool3d_3a_3x3
        net = tf.nn.max_pool3d(input=net, ksize=(1, 1, 3, 3, 1), strides=(1, 1, 2, 2, 1), padding=snt.SAME,
                               name='MaxPool3d_3a_3x3')
        # Mixed_3b
        net = self.inception_module(inputs=net, output_list=[64, 96, 128, 16, 32, 32], name='Mixed_3b',
                                    is_training=is_training)
        # Mixed_3c
        net = self.inception_module(inputs=net, output_list=[128, 128, 192, 32, 96, 64], name='Mixed_3c',
                                    is_training=is_training)

        # MaxPool3d_4a_3x3
        net = tf.nn.max_pool3d(input=net, ksize=(1, 3, 3, 3, 1), strides=(1, 2, 4, 4, 1), padding=snt.SAME,
                               name='MaxPool3d_4a_3x3')
        # Mixed_4b
        net = self.inception_module(inputs=net, output_list=[192, 96, 208, 16, 48, 64], name='Mixed_4b',
                                    is_training=is_training)
        # Mixed_4c
        net = self.inception_module(inputs=net, output_list=[160, 112, 224, 24, 64, 64], name='Mixed_4c',
                                    is_training=is_training)
        # Mixed_4d
        net = self.inception_module(inputs=net, output_list=[128, 128, 256, 24, 64, 64], name='Mixed_4d',
                                    is_training=is_training)
        # Mixed_4e
        net = self.inception_module(inputs=net, output_list=[112, 144, 288, 32, 64, 64], name='Mixed_4e',
                                    is_training=is_training)
        # Mixed_4f
        net = self.inception_module(inputs=net, output_list=[256, 160, 320, 32, 128, 128], name='Mixed_4f',
                                    is_training=is_training)

        # MaxPool3d_5a_3x3
        net = tf.nn.max_pool3d(input=net, ksize=(1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding=snt.SAME,
                               name='MaxPool3d_5a_3x3')
        # Mixed_5b
        net = self.inception_module(inputs=net, output_list=[256, 160, 320, 32, 128, 128], name='Mixed_5b',
                                    is_training=is_training)
        # # Mixed_5c
        net = self.inception_module(inputs=net, output_list=[384, 192, 384, 48, 128, 128], name='Mixed_5c',
                                    is_training=is_training)

        with tf.variable_scope('Logits'):
            net = tf.nn.avg_pool3d(net, ksize=(1, 2, 7, 7, 1), strides=(1, 1, 1, 1, 1), padding=snt.VALID,
                                   name='AvgPool3d_7x7')
            # dropout layer
            net = tf.nn.dropout(net, keep_prob=keep_prob)
            # logits layer
            logits = Unit3D(output_channels=self._num_classes,
                            kernel_shape=(1, 1, 1),
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv3d_0c_1x1')(net, is_training=is_training)

            # squeeze dimension
            if self._spatial_squeeze:
                logits = tf.squeeze(logits, axis=[2, 3], name='SpatialSqueeze')
            # average frame
            logits = tf.reduce_mean(logits, axis=1)


            return logits

    def inception_module(self, inputs, output_list, name=None, is_training=False):
        """
        inception module
        :param output_channel_list:
        :param name:
        :param is_training:
        :return:
        """
        with tf.variable_scope(name):
            # Branch_0
            with tf.variable_scope('Branch_0'):
                # Conv3d_0a_1x1
                branch_0 = Unit3D(output_channels=output_list[0],
                                  kernel_shape=(1, 1, 1),
                                  stride=(1, 1, 1),
                                  name='Conv3d_0a_1x1')(inputs, is_training=is_training)
            # Branch_1
            with tf.variable_scope('Branch_1'):
                # Conv3d_0a_1x1
                branch_1 = Unit3D(output_channels=output_list[1],
                                  kernel_shape=(1, 1, 1),
                                  stride=(1, 1, 1),
                                  name='Conv3d_0a_1x1')(inputs, is_training=is_training)
                # Conv3d_0b_3x3
                branch_1 = Unit3D(output_channels=output_list[2],
                                  kernel_shape=(3, 3, 3),
                                  stride=(1, 1, 1),
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
            # Branch_2
            with tf.variable_scope('Branch_2'):
                # Conv3d_0a_1x1
                branch_2 = Unit3D(output_channels=output_list[3],
                                  kernel_shape=(1, 1, 1),
                                  stride=(1, 1, 1),
                                  name='Conv3d_0a_1x1')(inputs, is_training=is_training)
                # Conv3d_0b_3x3
                branch_2 = Unit3D(output_channels=output_list[4],
                                  kernel_shape=(3, 3, 3),
                                  stride=(1, 1, 1),
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
            # Branch_3
            with tf.variable_scope('Branch_3'):
                # MaxPool3d_3x3
                branch_3 = tf.nn.max_pool3d(input=inputs, ksize=(1, 3, 3, 3, 1), strides=(1, 1, 1, 1, 1),
                                            padding=snt.SAME, name='MaxPool3d_0a_3x3')
                # Conv3d_0b_1x1
                branch_3 = Unit3D(output_channels=output_list[5],
                                  kernel_shape=(1, 1, 1),
                                  stride=(1, 1, 1),
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
            # concat
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=4)

            return net
