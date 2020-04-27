#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train_13d.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/27 上午11:17
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from DataProcess.read_video_tfrecord import get_num_samples, dataset_tfrecord, video_process
from I3D_Tensorflow.i3d_slim import I3D

# compatible GPU version problem
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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
flags.DEFINE_integer('batch_size', 16, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('epoch', 30, 'Number of epoch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('momentum_rate', 0.9, 'Initial momentum rate.')
flags.DEFINE_float('keep_prob', 0.8, 'Number of probability that each element is kept.')
flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')
flags.DEFINE_string('mode', 'Flow', 'train mode RGB|Flow.')
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
    train_per_epoch_step = int(np.ceil(train_num_samples / FLAGS.batch_size))
    val_per_epoch_step = int(np.ceil(val_num_samples / FLAGS.batch_size))

    # construct i3d network
    i3d = I3D(num_classes=FLAGS.num_classes,
              learning_rate=FLAGS.learning_rate,
              momentum_rate=FLAGS.momentum_rate,
              keep_prob=FLAGS.keep_prob,
              mode=FLAGS.mode)

    train_rgb_video, train_flow_video, train_label, train_filename = dataset_tfrecord(record_file=FLAGS.train_data,
                                                                                      batch_size=FLAGS.batch_size,
                                                                                      class_depth=FLAGS.num_classes,
                                                                                      epoch=FLAGS.epoch,
                                                                                      shuffle=True)

    val_rgb_video, val_flow_video, val_label, val_filename = dataset_tfrecord(record_file=FLAGS.val_data,
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


        # get variable of RGB scope
        # rgb and flow pretrain model
        if FLAGS.mode == 'RGB':
            model_path = os.path.join(pretrain_model_dir, 'rgb_imagenet', 'model.ckpt')
        elif FLAGS.mode == 'Flow':
            model_path = os.path.join(pretrain_model_dir, 'flow_imagenet', 'model.ckpt')

        summary_op = tf.summary.merge_all()
        # load pretrain model
        if FLAGS.is_pretrain:
            # remove variable of fc8 layer from pretrain model
            i3d.load_pretrain_model(sess, model_path, mode=FLAGS.mode, load_Logits=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if not coord.should_stop():
                # used to count the step per epoch

                train_acc = 0
                val_acc = 0

                for epoch in range(FLAGS.epoch):
                    print('Epoch: {0}/{1}'.format(epoch, FLAGS.epoch))

                    for step in range(train_per_epoch_step):

                        raw_rgb_video, raw_flow_video, input_label, input_filename = \
                            sess.run([train_rgb_video, train_flow_video, train_label, train_filename])

                        if FLAGS.mode == 'RGB':
                            video = video_process(raw_rgb_video, clip_size=6, target_shape=(224, 224), is_training=True)
                        elif FLAGS.mode == 'Flow':
                            video = video_process(raw_flow_video, clip_size=6, target_shape=(224, 224), is_training=True)
                        else:
                            print("Invalid mode")
                            raise IOError

                        input_video = sess.run(video)

                        feed_dict = i3d.fill_feed_dict(video_feed=input_video,
                                                       label_feed=input_label,
                                                       is_training=True)


                        _, train_loss, train_accuracy, summary = sess.run(fetches=[i3d.train, i3d.loss,
                                                                             i3d.accuracy, summary_op],
                                                                          feed_dict=feed_dict)
                        # # record the number of accuracy predict
                        # train_acc += np.sum(train_accuracy)
                        #
                        # # number of samples
                        # num_samples = (epoch * train_per_epoch_step + step + 1) * FLAGS.batch_size
                        # # calculate accuracy
                        # train_acc /= num_samples
                        # train_acc = np.sum(train_accuracy) / FLAGS.batch_size
                        train_acc = train_accuracy
                        print('step {0}: train loss: {1}, train accuracy: {2}'.
                              format(step, train_loss, train_acc))
                        write.add_summary(summary=summary, global_step=step)

                    for val_step in range(val_per_epoch_step):
                        raw_rgb_video, raw_flow_video, input_label, input_filename = \
                            sess.run([train_rgb_video, train_flow_video, train_label, train_filename])

                        if FLAGS.mode == 'RGB':
                            video = video_process(raw_rgb_video, clip_size=6, target_shape=(224, 224), is_training=False)
                        elif FLAGS.mode == 'Flow':
                            video = video_process(raw_flow_video, clip_size=6, target_shape=(224, 224), is_training=False)
                        else:
                            print("Invalid mode")
                            raise IOError

                        input_video = sess.run(video)

                        feed_dict = i3d.fill_feed_dict(video_feed=input_video,
                                                       label_feed=input_label,
                                                       is_training=False)
                        val_accuracy = sess.run(fetches=i3d.accuracy, feed_dict=feed_dict)

                        # record the number of accuracy predict
                        val_acc += val_accuracy

                    # number of samples
                    # num_samples = (epoch + 1) * FLAGS.batch_size

                    # # calculate accuracy
                    val_acc /= val_per_epoch_step
                    print('\t validation accuracy: {0}'.format(val_acc))

                write.close()

                # save model
                # get op name for save model
                rgb_input_op = i3d.input_data
                logit_op = i3d.logits.op
                # convert variable to constant
                input_graph_def = tf.get_default_graph().as_graph_def()
                constant_graph = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                              output_node_names=[rgb_input_op.op.name,
                                                                                                 logit_op.op.name])
                # save to serialize file
                with tf.gfile.FastGFile(name=model_name, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

        except Exception as e:
            print(e)
        coord.request_stop()
        coord.join(threads)
    sess.close()
    print('model training has complete')