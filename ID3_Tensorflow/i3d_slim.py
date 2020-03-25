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

class I3D():
  """
  I3D model
  """
  pass

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

  def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d'):
    """Initializes I3D model instance.

    Args:
      num_classes: The number of outputs in the logit layer (default 400, which
          matches the Kinetics dataset).
      spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
          before returning (default True).
      final_endpoint: The model contains many possible endpoints.
          `final_endpoint` specifies the last endpoint for the model to be built
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
      keep_prob: Probability for the tf.nn.dropout layer (float in
          [0, 1)).

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
    net = self.inception_module(inputs=net, output_list=[128, 128, 92, 32, 96, 64], name='Mixed_3b',
                                is_training=is_training)

    # MaxPool3d_4a_3x3
    net = tf.nn.max_pool3d(input=net, ksize=(1, 3, 3, 3, 1), strides=(1, 2, 2, 2, 1), padding=snt.SAME,
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
    # Mixed_5c
    net = self.inception_module(inputs=net, output_list=[384, 192, 384, 48, 128, 128], name='Mixed_5c',
                                is_training=is_training)

    with tf.variable_scope('Logits'):
      net = tf.nn.avg_pool3d(net, ksize=(1, 2, 7, 7, 1), strides=(1, 1, 1, 1, 1), padding=snt.VALID,
                             name='AvgPool3d_7x7')
      # dropout layer
      net = tf.nn.dropout(net, keep_prob=keep_prob)
      logits = Unit3D(output_channels=self._num_classes,
                      kernel_shape=(1, 1, 1),
                      activation_fn=None,
                      use_batch_norm=False,
                      use_bias=True,
                      name='Conv3d_0c_1x1')(net, is_training=is_training)
      # average frame
      logits = tf.reduce_mean(logits, axis=1)

      if self._spatial_squeeze:
        logits = tf.squeeze(logits, axis=[2, 3], name='SpatialSqueeze')

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















