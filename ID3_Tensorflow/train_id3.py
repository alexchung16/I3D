#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train_id3.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/11 下午3:31
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from DataProcess.read_video_tfrecord import get_num_samples, dataset_tfrecord
from ID3_Tensorflow.i3d_slim import I3D



original_dataset_dir = '/home/alex/Documents/dataset/video_binary'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecords')

train_data = os.path.join(tfrecord_dir, 'train')
val_data = os.path.join(tfrecord_dir, 'test')

model_path = os.path.join(os.getcwd(), 'model')
model_name = os.path.join(model_path, 'i3d.pb')
pretrain_model_dir = '/home/alex/Documents/pretraing_model/i3d'
logs_dir = os.path.join(os.getcwd(), 'logs')


flags = tf.app.flags
flags.DEFINE_integer('clips_size', 6, 'Number of clips size.')
flags.DEFINE_integer('height', 224, 'Number of height size.')
flags.DEFINE_integer('width', 225, 'Number of width size.')
flags.DEFINE_integer('depth', 3, 'Number of depth size.')
flags.DEFINE_integer('num_classes', 2, 'Number of image class.')
flags.DEFINE_integer('batch_size', 6, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('epoch', 30, 'Number of epoch size.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum_rate', 0.9, 'Initial momentum rate.')
flags.DEFINE_float('keep_prob', 0.8, 'Number of probability that each element is kept.')
flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')
flags.DEFINE_string('pretrain_model_dir', pretrain_model_dir, 'pretrain model dir.')
flags.DEFINE_string('train_data', train_data, 'Directory to put the training data.')
flags.DEFINE_string('val_data', val_data, 'Directory to put the validation data.')
flags.DEFINE_string('logs_dir', logs_dir, 'direct of summary logs.')
FLAGS = flags.FLAGS



if __name__ == "__main__":

    train_num_samples = get_num_samples(record_dir=FLAGS.train_data)
    val_num_samples = get_num_samples(record_dir=FLAGS.train_data)
    # approximate samples per epoch

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(train_num_samples / FLAGS.batch_size))
    val_batches_per_epoch = int(np.floor(val_num_samples / FLAGS.batch_size))

    # construct i3d network
    i3d = I3D(num_classes=FLAGS.num_classes,
              learning_rate=FLAGS.learning_rate,
              momentum_rate=FLAGS.momentum_rate,
              keep_prob=FLAGS.keep_prob,
              is_pretrain=FLAGS.is_pretrain)

    train_rgb_video, train_video, train_label, train_filename = dataset_tfrecord(record_file=FLAGS.train_data,
                                                                                batch_size=FLAGS.batch_size,
                                                                                class_depth=FLAGS.num_classes,
                                                                                epoch=FLAGS.epoch,
                                                                                shuffle=True)

    test_rgb_video, test_video, test_label, test_filename = dataset_tfrecord(record_file=FLAGS.val_data,
                                                                             batch_size=FLAGS.batch_size,
                                                                             class_depth=FLAGS.num_classes,
                                                                             epoch=FLAGS.epoch,
                                                                             shuffle=True)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)

        # graph
        graph = tf.get_default_graph()
        # write op
        write = tf.summary.FileWriter(logdir=FLAGS.logs_dir, graph=graph)

        model_variables = tf.trainable_variables()
        for var in model_variables:
            print(var.op.name)
            print(var.shape)

















