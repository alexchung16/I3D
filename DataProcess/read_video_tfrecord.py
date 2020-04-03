#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : read_video_tfrecord.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/26 下午6:45
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.python_io import tf_record_iterator

# original_dataset_dir = '/home/alex/Documents/datasets/dogs_vs_cat_separate'
original_dataset_dir = '/home/alex/Documents/dataset/video_binary'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecords')

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, 'test')


# def parse_example(serialized_sample, clip_size, target_shape, class_depth, is_training=False):
def parse_example(serialized_sample, class_depth, is_training=False):
    """
    parse tensor
    :param image_sample:
    :return:
    """

    # construct feature description
    image_feature_description ={

        "rgb_video": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "flow_video": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "height": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "width": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "rgb_depth": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "flow_depth": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "rgb_frames": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "flow_frames": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }
    feature = tf.io.parse_single_example(serialized=serialized_sample, features=image_feature_description)

    # parse feature
    rgb_video = tf.decode_raw(feature['rgb_video'], tf.uint8)
    flow_video = tf.decode_raw(feature['flow_video'], tf.float32)
    # shape = tf.cast(feature['shape'], tf.int32)
    height = tf.cast(feature['height'], tf.int32)
    width = tf.cast(feature['width'], tf.int32)
    rgb_depth = tf.cast(feature['rgb_depth'], tf.int32)
    flow_depth = tf.cast(feature['flow_depth'], tf.int32)
    rgb_frames = tf.cast(feature['rgb_frames'], tf.int32)
    flow_frames = tf.cast(feature['flow_frames'], tf.int32)
    label = tf.cast(feature['label'], tf.int32)

    rgb_video = tf.reshape(rgb_video, [rgb_frames, height, width, rgb_depth])
    flow_video = tf.reshape(flow_video, [flow_frames, height, width, flow_depth])

    filename = tf.cast(feature['filename'], tf.string)
    # resize image shape
    # random crop image
    # before use shuffle_batch, use random_crop to make image shape to special size
    # first step enlarge image size
    # second step dataset operation

    # # image augmentation
    # rgb_video = video_process(input_video=rgb_video, clip_size=clip_size, target_shape=target_shape, mode='rgb',
    #                           is_training=is_training)
    # flow_video = video_process(input_video=flow_video, clip_size=clip_size, target_shape=target_shape, mode='flow',
    #                            is_training=is_training)
    # onehot label
    label = tf.one_hot(indices=label, depth=class_depth)

    return rgb_video, flow_video, label, filename


def video_process(input_video_batch, clip_size, target_shape, mode='rgb', is_training=False):
    """

    :param input_video:
    :param clip_size:
    :param target_shape:
    :param model:
    :param is_training:
    :return:
    """

    # enlarge image to same size
    # squeeze batch dimension

    input_video_batch = tf.convert_to_tensor(input_video_batch)
    try:
        batch_size = int(input_video_batch.get_shape()[0])

        output_video_batch = None
        for index in range(batch_size):
            # process video
            output_video = augmentation_video(video=tf.gather(input_video_batch, indices=index, axis=0),
                                              clip_size=clip_size,
                                              target_shape=target_shape,
                                              mode=mode,
                                              is_training=is_training)
            # expend video dim for concat to video batch
            output_video = tf.expand_dims(output_video, axis=0)
            if output_video_batch is None:
                output_video_batch = output_video
            else:
                output_video_batch = tf.concat(values=[output_video_batch, output_video], axis=0)

        return output_video_batch

    except Exception as e:

        print(e)

def augmentation_video(video, clip_size, target_shape, mode, is_training=False):
    """

    :param video:
    :param target_shape:
    :param mode:
    :param is_training:
    :return:
    """

    # frames = tf.to_float(frames)
    # clip_size = tf.convert_to_tensor(clip_size, dtype=tf.float32)
    # start_frame, end_frame = tf.cond(tf.greater(frames, clip_size),
    #                                  true_fn=
    #                                  lambda : (tf.random.uniform(shape=[], minval=0, maxval=(frames - clip_size -1)),
    #                                            start_frame +  clip_size),
    #                                  false_fn= lambda : (0, start_frame + frames))
    shape = video.get_shape()
    frames, height, width, depth = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])

    if clip_size < frames:
        start_frame = np.random.randint(low=0, high=(frames - clip_size - 1), dtype=np.int32)
        end_frame = start_frame + clip_size
    else:
        start_frame = 0
        end_frame = start_frame + frames

    output_video = None
    for clip_index in range(start_frame, end_frame, 1):

        frame = tf.gather(params=video, indices=clip_index, axis=0)

        frame = augmentation_image(frame, target_shape=target_shape, mode=mode, is_training=is_training)

        # expend dimension
        frame = tf.expand_dims(frame, axis=0)

        if output_video is None:
            output_video = frame
        else:
            output_video = tf.concat(values=[output_video, frame], axis=0)

    # # recover to input dimension
    # output_video = tf.expand_dims(output_video, axis=0)

    return output_video


def augmentation_image(image, target_shape, mode='rgb', is_training=False):
    """
    frame augmentation
    :param image:
    :param target_shape:
    :param is_training:
    :return:
    """
    # resize
    image = aspect_preserve_resize(image, resize_side_min=np.rint(target_shape[0] * 1.04),
                                   resize_side_max=np.rint(target_shape[0] * 2.08), is_training=is_training)

    #  crop to target_size
    image = image_crop(image, output_height=target_shape[0], output_width=target_shape[1],
                       is_training=is_training)
    if is_training:
        image = tf.image.flip_left_right(image)

    # transfer pixel size to (0., 1.)
    # if mode == 'rgb':
    #     image = tf.divide(tf.cast(image, dtype=tf.float32), 255.)
    #     # [0, 1] => [-0.5, 0.5]
    #     image = tf.subtract(image, 0.5)
    #     # [-0.5, 0.5] => [-1.0, 1.0]
    #     image = tf.multiply(image, 2.0)
    # elif mode == 'flow':
    #     pass

    return image


def aspect_preserve_resize(image, resize_side_min=256, resize_side_max=512, is_training=False):
    """

    :param image_tensor:
    :param output_height:
    :param output_width:
    :param resize_side_min:
    :param resize_side_max:
    :return:
    """
    if is_training:
        smaller_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max, dtype=tf.float32)
    else:
        smaller_side = resize_side_min

    shape = tf.shape(image)

    height, width = tf.to_float(shape[0]), tf.to_float(shape[1])

    resize_scale = tf.cond(pred=tf.greater(height, width),
                           true_fn=lambda : smaller_side / width,
                           false_fn=lambda : smaller_side / height)

    new_height = tf.to_int32(tf.rint(height * resize_scale))
    new_width = tf.to_int32(tf.rint(width * resize_scale))

    resize_image = tf.image.resize(image, size=(new_height, new_width))

    # output type as input type
    resize_image = tf.cast(resize_image, dtype=image.dtype)

    return resize_image


def image_crop(image, output_height=224, output_width=224, is_training=False):
    """

    :param image:
    :param output_height:
    :param output_width:
    :param is_training:
    :return:
    """
    shape = tf.shape(image)
    depth = shape[2]
    if is_training:

        crop_image = tf.image.random_crop(image, size=(output_height, output_width, depth))
    else:
        crop_image = central_crop(image, output_height, output_width)

        # output type as input type
    crop_image = tf.cast(crop_image, dtype=image.dtype)

    return crop_image

def central_crop(image, crop_height=224, crop_width=224):
    """
    image central crop
    :param image:
    :param output_height:
    :param output_width:
    :return:
    """

    shape = tf.shape(image)
    height, width, depth = shape[0], shape[1], shape[2]

    # calculate offset width and height for clip operation
    offset_height = (height - crop_height) / 2
    offset_width = (width - crop_width) / 2

    # assert image rank must be 3
    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal 3'])

    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, depth])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(height, crop_height),
            tf.greater_equal(width, crop_width)),
        ['Image size greater than the crop size'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    with tf.control_dependencies([size_assertion]):
        # crop with slice
        crop_image = tf.slice(image, begin=offsets, size=cropped_shape)

    return tf.reshape(crop_image, cropped_shape)

def dataset_tfrecord(record_file, class_depth, epoch=5, batch_size=10, shuffle=True, is_training=False):
    """
    construct iterator to read image
    :param record_file:
    :return:
    """
    record_list = []
    # check record file format
    if os.path.isfile(record_file):
        record_list = [record_file]
    else:
        for filename in os.listdir(record_file):
            record_list.append(os.path.join(record_file, filename))
    # # use dataset read record file
    raw_img_dataset = tf.data.TFRecordDataset(record_list)
    # execute parse function to get dataset
    # This transformation applies map_func to each element of this dataset,
    # and returns a new dataset containing the transformed elements, in the
    # same order as they appeared in the input.
    # when parse_example has only one parameter (office recommend)
    # parse_img_dataset = raw_img_dataset.map(parse_example)
    # when parse_example has more than one parameter which used to process data
    parse_img_dataset = raw_img_dataset.map(lambda series_record:
                                            parse_example(series_record, class_depth, is_training=is_training))
    # get dataset batch
    if shuffle:
        shuffle_batch_dataset = parse_img_dataset.shuffle(buffer_size=batch_size*4).repeat(epoch).batch(batch_size=batch_size)
    else:
        shuffle_batch_dataset = parse_img_dataset.repeat(epoch).batch(batch_size=batch_size)
    # make dataset iterator
    rgb_video, flow_video, label, filename = shuffle_batch_dataset.make_one_shot_iterator().get_next()


    return rgb_video, flow_video, label, filename


def get_num_samples(record_dir):
    """
    get tfrecord numbers
    :param record_file:
    :return:
    """

    record_list = []
    # check record file format

    for filename in os.listdir(record_dir):
        record_list.append(os.path.join(record_dir, filename))

    num_samples = 0
    for record_file in record_list:
        for record in tf_record_iterator(record_file):
            num_samples += 1
    return num_samples

if __name__ == "__main__":
    record_file = os.path.join(tfrecord_dir, 'train')
    rgb_video_batch, flow_video_batch, label_batch, filename = dataset_tfrecord(record_file=record_file,
                                                                                class_depth=5,
                                                                                batch_size=6,
                                                                                is_training=True)
    # augmentation video
    # create local and global variables initializer group
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)

        num_samples = get_num_samples(record_file)
        print('all sample size is {0}'.format(num_samples))
        # create Coordinator to manage the life period of multiple thread
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph to execute input queue operation
        # the step contain two operation:filename to filename queue and sample to sample queue
        threads = tf.train.start_queue_runners(coord=coord)
        print('threads: {0}'.format(threads))
        try:
            if not coord.should_stop():
                raw_rgb_video, raw_flow_video, label = sess.run([rgb_video_batch, flow_video_batch, label_batch])

                rgb_video = video_process(raw_rgb_video, clip_size=6, target_shape=(224, 224), is_training=True)
                flow_video = video_process(raw_flow_video, clip_size=6, target_shape=(224, 224), is_training=True)

                rgb_video, flow_video = sess.run([rgb_video, flow_video])

                plt.imshow(rgb_video[0][0])
                print(flow_video[0][0])
                plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()

        # waiting all threads safely exit
        coord.join(threads)
        sess.close()
