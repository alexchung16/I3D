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

from DataProcess.read_video_tfrecord import get_num_samples, dataset_tfrecord, video_process
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
    val_num_samples = get_num_samples(record_dir=FLAGS.val_data)
    # approximate samples per epoch

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.ceil(train_num_samples / FLAGS.batch_size))
    val_batches_per_epoch = int(np.ceil(val_num_samples / FLAGS.batch_size))

    # construct i3d network
    i3d = I3D(num_classes=FLAGS.num_classes,
              learning_rate=FLAGS.learning_rate,
              momentum_rate=FLAGS.momentum_rate,
              keep_prob=FLAGS.keep_prob,
              is_pretrain=FLAGS.is_pretrain)

    train_rgb_video, train_flow_video, train_label, train_filename = dataset_tfrecord(record_file=FLAGS.train_data,
                                                                                      batch_size=FLAGS.batch_size,
                                                                                      class_depth=FLAGS.num_classes,
                                                                                      epoch=FLAGS.epoch,
                                                                                      shuffle=True)

    test_rgb_video, test_flow_video, test_label, test_filename = dataset_tfrecord(record_file=FLAGS.val_data,
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
        rgb_model_path = os.path.join(pretrain_model_dir, 'rgb_imagenet', 'model.ckpt')
        flow_model_path = os.path.join(pretrain_model_dir, 'flow_imagenet', 'model.ckpt')

        summary_op = tf.summary.merge_all()
        # load pretrain model
        if FLAGS.is_pretrain:
            # remove variable of fc8 layer from pretrain model
            i3d.load_pretrain_model(sess, rgb_model_path, flow_model_path, load_Logits=False)

        # print(sess.run('vgg_16/conv1/conv1_1/biases:0'))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if not coord.should_stop():
                # used to count the step per epoch


                train_rgb_acc = 0
                train_flow_acc = 0
                train_model_acc = 0

                val_rgb_acc = 0
                val_flow_acc = 0
                val_model_acc = 0

                for epoch in range(FLAGS.epoch):
                    print('Epoch: {0}/{1}'.format(epoch, FLAGS.epoch))

                    for train_step in range(train_batches_per_epoch):

                        raw_rgb_video, raw_flow_video, train_label, train_filename = \
                            sess.run([train_rgb_video, train_flow_video, train_label, train_filename])

                        rgb_video = video_process(raw_rgb_video, clip_size=6, target_shape=(224, 224), is_training=True)
                        flow_video = video_process(raw_flow_video, clip_size=6, target_shape=(224, 224), is_training=True)

                        input_rgb_video, input_flow_video = sess.run([rgb_video, flow_video])

                        feed_dict = i3d.fill_feed_dict(rgb_video_feed=input_flow_video,
                                                       flow_video_feed=input_flow_video,
                                                       label_feed=train_label)

                        _, rgb_loss, flow_loss, train_accuracy, summary = sess.run(fetches=[i3d.train, i3d.rgb_loss,
                                                                                            i3d.flow_loss, i3d.accuracy,
                                                                                            summary_op],
                                                                                   feed_dict=feed_dict)
                        train_rgb_acc += np.sum(train_accuracy[0])
                        train_flow_acc += np.sum(train_accuracy[1])
                        train_model_acc += np.sum(train_accuracy[2])

                        print('step {0}: train rgb loss: {1}, train flow loss: {2} train rgb accuracy: {3}, '
                              'train flow accuracy {4}, model accuracy {5}'.format(train_step, rgb_loss, flow_loss,
                                                                                   train_rgb_acc, train_flow_acc,
                                                                                   train_model_acc))
                        write.add_summary(summary=summary, global_step=train_step)

                    # for val_step in range(val_batches_per_epoch):
                    #     raw_rgb_video, raw_flow_video, train_label, train_filename = \
                    #         sess.run([train_rgb_video, train_flow_video, train_label, train_filename])
                    #
                    #     rgb_video = video_process(raw_rgb_video, clip_size=6, target_shape=(224, 224))
                    #     flow_video = video_process(raw_flow_video, clip_size=6, target_shape=(224, 224),
                    #                                is_training=True)
                    #
                    #     input_rgb_video, input_flow_video = sess.run([rgb_video, flow_video])
                    #
                    #     feed_dict = i3d.fill_feed_dict(rgb_video_feed=input_flow_video,
                    #                                    flow_video_feed=input_flow_video,
                    #                                    label_feed=train_label)
                    #     accuracy = sess.run(fetches=i3d.accuracy,
                    #                         feed_dict=feed_dict)
                    #
                    # val_acc = val_acc / val_batches_per_epoch
                    #
                    # print('epoch{0}: val accuracy value {1}'.format(epoch, val_acc))

                write.close()

                # save model
                # get op name for save model
                input_op = alex_net.raw_input_data.name
                logit_op = alex_net.logits.name
                # convert variable to constant
                input_graph_def = tf.get_default_graph().as_graph_def()
                constant_graph = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                              [input_op.split(':')[0],
                                                                               logit_op.split(':')[0]])
                # save to serialize file
                with tf.gfile.FastGFile(name=model_name, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

        except Exception as e:
            print(e)
        coord.request_stop()
        coord.join(threads)
    sess.close()
    print('model training has complete')


















































