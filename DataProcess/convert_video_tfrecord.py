#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : convert_video_tfrecord.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/26 下午3:51
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import numpy as np
import shutil
import cv2 as cv
import tensorflow as tf
from multiprocessing import Pool

from Util.tools import view_bar, make_dir, refresh_dir

video_dir = '/home/alex/Documents/dataset/video_binary'
dataset_dir = os.path.join(video_dir, 'split_bike_raft')

rgb_record_dir = os.path.join(video_dir, 'tfrecords', 'rgb')
flow_record_dir = os.path.join(video_dir, 'tfrecords', 'flow')

train_data_dir = os.path.join(dataset_dir, 'train')
val_data_dir = os.path.join(dataset_dir, 'val')

rgb_train_target_dir = os.path.join(rgb_record_dir, 'train')
rgb_val_target_dir = os.path.join(rgb_record_dir, 'val')

flow_train_target_dir = os.path.join(flow_record_dir, 'train')
flow_val_target_dir = os.path.join(flow_record_dir, 'val')

flags = tf.app.flags
flags.DEFINE_string('dataset_dir', dataset_dir, 'dataset dir')
flags.DEFINE_string('train_data_dir', train_data_dir, 'train dataset dir')
flags.DEFINE_string('val_data_dir', val_data_dir, 'validation dataset dir')
flags.DEFINE_string('rgb_train_target_dir', rgb_train_target_dir, 'train target dir')
flags.DEFINE_string('rgb_val_target_dir', rgb_val_target_dir, 'validation target dir')
flags.DEFINE_string('flow_train_target_dir', flow_train_target_dir, 'train target dir')
flags.DEFINE_string('flow_val_target_dir', flow_val_target_dir, 'validation target dir')
flags.DEFINE_string('save_name', 'train', 'save name')
FLAGS = flags.FLAGS


# ----------------------------------get rgb frames and optical flow frames-------------------------------------
def get_video_length(video_path):
    """
    size of video frame
    :param video_path:
    :return:
    """
    cap = cv.VideoCapture(video_path)

    # check the video available
    if not cap.isOpened():
        # raise ValueError('Could not open the file.\n{0}'.format(video_path))
        print('Could not open the file.\n{0}'.format(video_path))

    CAP_PROP_FRAME_COUNT = cv.CAP_PROP_FRAME_COUNT
    length = int(cap.get(CAP_PROP_FRAME_COUNT))

    return length


def get_rgb_flow(video_path,  sample_frames = None, epsilon=1e-5):
    """
    compute optical flow with DualTVL1 algorithm
    :param video_path:
    :return:
    """
    # save rgb and flow
    rgb_videos = []
    flow_videos = []

    # get size of video frames
    video_length = get_video_length(video_path)

    if sample_frames is not None and sample_frames<=video_length:
        start_frame  = np.random.randint(0, (video_length - sample_frames))
        end_frame = start_frame + sample_frames
    else:
        start_frame = 0
        end_frame = video_length


    TVL1 = cv.optflow.DualTVL1OpticalFlow_create()

    cap = cv.VideoCapture(video_path)

    pre_ret, pre_frame = cap.read()

    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)

    for index in range(end_frame):

        cur_ret, cur_frame = cap.read()
        cur_gray = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

        if index < start_frame:
            continue
        else:
            # convert image channel from BGR to RGB
            rgb_videos.append(cv.cvtColor(pre_frame, cv.COLOR_BGR2RGB))
            # calculate optical flow
            cur_flow = TVL1.calc(pre_gray, cur_gray, None)
            assert cur_flow.dtype == np.float32
            # truncate [-20, 20]
            cur_flow[cur_flow > 20] = 20
            cur_flow[cur_flow < -20] =-20
            # scale to [-1, 1]
            max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
            cur_flow = cur_flow / (max_val(cur_flow) + epsilon)
            flow_videos.append(cur_flow)
        pre_frame, pre_gray = cur_frame, cur_gray
    cap.release()

    return np.array(rgb_videos), np.array(flow_videos)


def rgb_frame_extract(video_path,  sample_frames=None):
    """
    extract rgb frame
    :param video_path:
    :param sample_frames:
    :param epsilon:
    :return:
    """
    # save rgb frame
    rgb_videos = []

    # get size of video frames
    video_length = get_video_length(video_path)

    if sample_frames is not None and sample_frames<=video_length:
        start_frame = np.random.randint(0, (video_length - sample_frames))
        end_frame = start_frame + sample_frames
    else:
        start_frame = 0
        end_frame = video_length

    cap = cv.VideoCapture(video_path)
    for index in range(end_frame):
        # capture frame
        cur_ret, cur_frame = cap.read()

        if index < start_frame:
            continue
        else:
            # convert image channel from BGR to RGB
            rgb_videos.append(cv.cvtColor(cur_frame, cv.COLOR_BGR2RGB))
    cap.release()

    return np.array(rgb_videos)


def flow_frame_extract(video_path,  sample_frames = None, epsilon=1e-5):
    """
    extract optical flow frame
    :param video_path:
    :param sample_frames:
    :param epsilon:
    :return:
    """
    # save flow frame
    flow_videos = []

    # get size of video frames
    video_length = get_video_length(video_path)

    if sample_frames is not None and sample_frames <= (video_length-2):
        start_frame = np.random.randint(0, (video_length - sample_frames - 2))
        end_frame = start_frame + sample_frames
    else:
        start_frame = 0
        end_frame = video_length - 2

    TVL1 = cv.optflow.DualTVL1OpticalFlow_create()

    cap = cv.VideoCapture(video_path)

    pre_ret, pre_frame = cap.read()

    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)

    for index in range(end_frame):

        cur_ret, cur_frame = cap.read()
        cur_gray = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

        if index < start_frame:
            continue
        else:
            # calculate optical flow
            cur_flow = TVL1.calc(pre_gray, cur_gray, None)
            assert cur_flow.dtype == np.float32
            # truncate [-20, 20]
            cur_flow[cur_flow > 20] = 20
            cur_flow[cur_flow < -20] = -20
            # scale to [-1, 1]
            max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
            cur_flow = cur_flow / (max_val(cur_flow) + epsilon)
            flow_videos.append(cur_flow)
        pre_frame, pre_gray = cur_frame, cur_gray
    cap.release()

    return  np.array(flow_videos)


# -----------------------------------------convert arg and flow to tfrecord--------------------------------------
def execute_convert_tfrecord(data_path, target_path, sample_frames=None,
                             per_record_capacity=500, shuffle=True, mode=None):
    """

    :param source_path:
    :param outputs_path:
    :param split_ratio:
    :param shuffle:
    :return:
    """
    # refresh  tfrecord path
    refresh_dir(target_path)

    video_names, video_labels, train_classes_map = get_video_label_info(data_path, shuffle=shuffle)

    video_to_record(save_path=target_path,
                    video_names=video_names,
                    sample_frames=sample_frames,
                    video_labels=video_labels,
                    record_capacity=per_record_capacity,
                    mode=mode)

    return True

def video_to_record(save_path, video_names, video_labels=None, sample_frames=None, record_capacity=500, mode=None):
    """

    :param save_path:
    :param video_name_list:
    :param sample_frames:
    :param labels_list:
    :param record_capacity:
    :param mode: None|rgb|flow
    :return:
    """

    remainder_num = len(video_names) % record_capacity
    if remainder_num == 0:
        num_record = int(len(video_labels) / record_capacity)
    else:
        num_record = int(len(video_labels) / record_capacity) + 1

    count = 1
    for index in range(num_record):
        record_filename = os.path.join(save_path, 'tfrecord-{0}.record'.format(index))
        writer = tf.io.TFRecordWriter(record_filename)
        if index < num_record - 1:
            sub_video_names = video_names[index * record_capacity: (index + 1) * record_capacity]
            sub_video_labels = video_labels[index * record_capacity: (index + 1) * record_capacity]
        else:
            sub_video_names = video_names[(index * record_capacity): (index * record_capacity + remainder_num)]
            sub_video_labels = video_labels[(index * record_capacity): (index * record_capacity + remainder_num)]

        for video_name, label in zip(sub_video_names, sub_video_labels):
            try:
                video_record = None
                # synchronization to write rgb and flow data
                if mode is None:
                    rgb_video, flow_video =  get_rgb_flow(video_name, sample_frames)

                    rgb_frames = rgb_video.shape[0]
                    flow_frames = flow_video.shape[0]
                    height = rgb_video.shape[1]
                    width = rgb_video.shape[2]
                    rgb_depth = rgb_video.shape[3]
                    flow_depth = flow_video.shape[3]

                    video_record = serialize_example(rgb_video, flow_video,  label, rgb_frames, flow_frames, height, width,
                                                     rgb_depth, flow_depth, video_name, mode=mode)

                elif mode == "rgb":
                    rgb_video = rgb_frame_extract(video_name, sample_frames)
                    rgb_frames = rgb_video.shape[0]
                    rgb_height = rgb_video.shape[1]
                    rgb_width = rgb_video.shape[2]
                    rgb_depth = rgb_video.shape[3]

                    video_record = serialize_example(rgb_video=rgb_video, label=label, rgb_frames=rgb_frames,
                                                     frame_height=rgb_height, frame_width=rgb_width, rgb_depth=rgb_depth,
                                                     filename=video_name, mode=mode)

                elif mode == 'flow':
                    flow_video = flow_frame_extract(video_name, sample_frames)
                    flow_frames = flow_video.shape[0]
                    flow_height = flow_video.shape[1]
                    flow_width = flow_video.shape[2]
                    flow_depth = flow_video.shape[3]

                    video_record = serialize_example(flow_video=flow_video, label=label, flow_frames=flow_frames,
                                                     frame_height=flow_height, frame_width=flow_width,
                                                     flow_depth=flow_depth, filename=video_name, mode=mode)
                writer.write(record=video_record)
                view_bar(message='Conversion progress', num=count, total=len(video_names))
                count += 1

            except Exception as e:
                print('\nFailed convert {0} , Please Check the samples whether exist or correct format'.format(video_name))
                continue

    print("\nThere are {0} samples has successfully convert to tfrecord, save at {1}".format(count, save_path))

    return True


def get_video_label_info(data_path, classes_map=None, shuffle=True):
    """
    get image list and label list
    :param data_path:
    :return:
    """

    video_names = []  # image name
    video_labels = []  # image label
    class_map = {}

    if classes_map is None:
        # classes name
        for subdir in sorted(os.listdir(data_path)):
            if os.path.isdir(os.path.join(data_path, subdir)):
                class_map[subdir] = len(class_map)
    else:
        class_map = classes_map

    for class_name, class_label in class_map.items():
        # get image file each of class
        class_dir = os.path.join(data_path, class_name)
        video_list = os.listdir(class_dir)

        for index, video_name in enumerate(video_list):
            video_names.append(os.path.join(class_dir, video_name))
            video_labels.append(class_label)

    num_samples = len(video_names)

    if shuffle:
        video_names_shuffle = []
        video_labels_shuffle = []
        index_array = np.random.permutation(num_samples)

        for i, index in enumerate(index_array):
            video_names_shuffle.append(video_names[index])
            video_labels_shuffle.append(video_labels[index])

        video_names = video_names_shuffle
        video_labels = video_labels_shuffle

    return video_names, video_labels, class_map


def serialize_example(rgb_video=None, flow_video=None,  label=None, rgb_frames=None, flow_frames=None, frame_height=None,
                      frame_width=None, rgb_depth=None, flow_depth=None,  filename=None, mode=None):
    """
    create a tf.Example message to be written to a file
    :param rgb_video:
    :param flow_video:
    :param label:
    :param rgb_frames:
    :param flow_frames:
    :param frame_height:
    :param frame_width:
    :param rgb_depth:
    :param flow_depth:
    :param filename:
    :param mode:
    :return:
    """
    # create a dict mapping the feature name to the tf.Example compatible
    # image_shape = tf.image.decode_jpeg(image_string).eval().shape
    feature = None
    # synchronization save rgb and optical flow
    if mode is None:
        feature = {
            "rgb_video": _bytes_feature(rgb_video.tobytes()),
            "flow_video": _bytes_feature(flow_video.tobytes()),
            "label": _int64_feature(label),
            "height": _int64_feature(frame_height),
            "width": _int64_feature(frame_width),
            "rgb_depth": _int64_feature(rgb_depth),
            "flow_depth": _int64_feature(flow_depth),
            "rgb_frames": _int64_feature(rgb_frames),
            "flow_frames": _int64_feature(flow_frames),
            "filename": _bytes_feature(filename.encode())
        }
    # only rgb serialize
    elif mode == 'rgb':
        feature = {
            "rgb_video": _bytes_feature(rgb_video.tobytes()),
            "label": _int64_feature(label),
            "height": _int64_feature(frame_height),
            "width": _int64_feature(frame_width),
            "flow_depth": _int64_feature(rgb_depth),
            "flow_frames": _int64_feature(rgb_frames),
            "filename": _bytes_feature(filename.encode())
        }
    # only flow serialize
    elif mode == 'flow':
        feature = {
            "flow_video": _bytes_feature(flow_video.tobytes()),
            "label": _int64_feature(label),
            "height": _int64_feature(frame_height),
            "width": _int64_feature(frame_width),
            "flow_depth": _int64_feature(flow_depth),
            "flow_frames": _int64_feature(flow_frames),
            "filename": _bytes_feature(filename.encode())
        }
    # create a feature message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":

    # def execute_tfrecord(args):
    #     source_path, outputs_path = args
    #     execute_convert_tfrecord(source_path, outputs_path)
    #
    #     return True
    # pool = Pool(2)
    # pool.map(execute_tfrecord, zip(FLAGS.dataset_dir, FLAGS.save_dir))

    # execute train dataset to tfrecord
    execute_convert_tfrecord(data_path=FLAGS.train_data_dir, target_path=FLAGS.flow_train_target_dir, sample_frames=6,
                             per_record_capacity=100, mode='flow')
    execute_convert_tfrecord(data_path=FLAGS.val_data_dir, target_path=FLAGS.flow_val_target_dir, sample_frames=6,
                             per_record_capacity=100, mode='flow')