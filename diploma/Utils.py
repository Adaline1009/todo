import keras
import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)

import os
from os.path import join
import json

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.models import Model, load_model
from collections import Counter
import cv2


def get_counter(dirpath, tag):
    dirname = os.path.basename(dirpath)
    ann_dirpath = join(dirpath, 'ann')
    letters = ''
    lens = []
    for filename in os.listdir(ann_dirpath):
        json_filepath = join(ann_dirpath, filename)
        ann = json.load(open(json_filepath, 'r'))
        #  tags = ann['tags']
        #  if tag in tags:
        description = ann['description']
        lens.append(len(description))
        letters += description
    print('Max plate length in "%s":' % dirname, max(Counter(lens).keys()))
    return Counter(letters)


def decode_batch(out, letters):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret


def build_data(img_filepath, img_w, img_h):
    img = cv2.imread(img_filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_w, img_h))
    img = img.astype(np.float32)
    img /= 255
    return img;


def text_to_labels(text, letters):
    return list(map(lambda x: letters.index(x), text))

def labels_to_text(labels, letters):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def next_batch(batch_size, img_w, img_h, max_text_len, letters, downsample_factor, img):
    # width and height are backwards from typical Keras convention
    # because width is the time dimension when it gets fed into the RNN
    if K.image_data_format() == 'channels_first':
        X_data = np.ones([1, 1, img_w, img_h])
    else:
        X_data = np.ones([1, img_w, img_h, 1])
    Y_data = np.ones([1, max_text_len])
    input_length = np.ones((batch_size, 1)) * (img_w // downsample_factor - 2)
    label_length = np.zeros((batch_size, 1))
    source_str = []
    #img = np.reshape(img, [img.shape[1], img.shape[0]])
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    X_data[0] = img
    # Y_data[0] = text_to_labels(text, letters)
    # source_str.append(text)
    # label_length[0] = len(text)
    inputs = {
        'the_input': X_data,
        'the_labels': Y_data,
        'input_length': input_length,
        'label_length': label_length,
        # 'source_str': source_str
    }
    outputs = {'ctc': np.zeros([batch_size])}
    yield (inputs, outputs)