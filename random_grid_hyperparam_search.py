import numpy as np
import os
import sys
import tarfile
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image
import math
import random
import re
import scipy.io
import PIL
from numpy import *
from pylab import *
from PIL import Image
from collections import defaultdict
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_EPOCHS = 10
BATCH_SIZE = 40
VAL_BATCH_SIZE = 5

ABS_PATH = '/home/timaiken/final_project/' #'/Users/timaiken/Stanford/CS231N/final_project/'

net_data = np.load('tf_files/bvlc_alexnet.npy', encoding='latin1').item()
metadata = np.load('tf_files/metadata.npy').item()

out_pool_size = [8, 6, 4]
hidden_dim = 0
for item in out_pool_size:
  hidden_dim = hidden_dim + item * item

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding = "VALID", group = 1):
  '''From https://github.com/ethereon/caffe-tensorflow
  '''
  c_i = input.get_shape()[-1]
  assert c_i % group == 0
  assert c_o % group == 0
  convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

  if group == 1:
    conv = convolve(input, kernel)
  else:
    input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
    kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    conv = tf.concat(axis=3, values=output_groups)
  return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.01, name=name)
  return tf.Variable(initial)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape, name=name)
  return tf.Variable(initial)

def conv2d(x, W, stride_h, stride_w, padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride_h, stride_w, 1], padding=padding)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Spatial Pyramid Pooling block
# https://arxiv.org/abs/1406.4729
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
  """
  previous_conv: a tensor vector of previous convolution layer
  num_sample: an int number of image in the batch
  previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
  out_pool_size: a int vector of expected output size of max pooling layer
  
  returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
  """
  for i in range(len(out_pool_size)):
    h_strd = h_size = math.ceil(float(previous_conv_size[0]) / out_pool_size[i])
    w_strd = w_size = math.ceil(float(previous_conv_size[1]) / out_pool_size[i])
    pad_h = int(out_pool_size[i] * h_size - previous_conv_size[0])
    pad_w = int(out_pool_size[i] * w_size - previous_conv_size[1])
    new_previous_conv = tf.pad(previous_conv, tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]]))
    max_pool = tf.nn.max_pool(new_previous_conv,
                   ksize=[1,h_size, h_size, 1],
                   strides=[1,h_strd, w_strd,1],
                   padding='SAME')
    if (i == 0):
      spp = tf.reshape(max_pool, [num_sample, -1])
    else:
      spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [num_sample, -1])])
  
  return spp

train_files = [filename for filename in os.listdir('tf_files') if filename.startswith('training')]
val_files = [filename for filename in os.listdir('tf_files') if filename.startswith('val')]

rmses = []
validation_accuracies = []
x_range = []
startedModel = False
print('Training ...\n')

# Training block
# 1. Combime all iamges have the same size to a batch.
# 2. Then, train parameters in a batch
# 3. Transfer trained parameters to another batch

L_RATES = 10**(-1 * np.random.uniform(1.0, 5.0, 15))
W_DECAYS = 10**(-1 * np.random.uniform(0.0, 5.0, 15))
DROPOUTS = np.random.uniform(0.0, 1.0, 15)

best_val = 1000000
best_l = 0
best_w = 0
best_d = 0

for i in range(15):
  DROPOUT = DROPOUTS[i]
  LEARNING_RATE  = L_RATES[i]
  WEIGHT_DECAY = W_DECAYS[i]
  print("**Testing hyperparameters**")
  print("Learning rate:", LEARNING_RATE)
  print("Dropout:", DROPOUT)
  print("Weight Decay:", WEIGHT_DECAY, "\n")
  sys.stdout.flush()

  for it in range(NUM_EPOCHS):
    print("Epoch:", it)

    total_train_rms = 0
    total_train_count = 0

    for train_f in train_files:

      split = train_f.split("_")
      w = split[1]
      h = split[2].split(".")[0]

      label = train_f.split(".")[0] + '/label' 
      image = train_f.split(".")[0] + '/image' 

      graph = tf.Graph()
      with graph.as_default():

        feature = {label: tf.FixedLenFeature([], tf.float32), image: tf.FixedLenFeature([], tf.string)}

        filename_queue = tf.train.string_input_producer([ABS_PATH + '/tf_files/' + train_f])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features[image], tf.float64)
        label = tf.cast(features[label], tf.float32)
        label = tf.reshape(label, [-1])
        image = tf.reshape(image, [int(w), int(h), 3])
        images, labels = tf.train.batch([image, label], \
                 batch_size=BATCH_SIZE, \
                 capacity=BATCH_SIZE, \
                 num_threads=8, \
                 allow_smaller_final_batch=False)
        x = tf.placeholder('float', shape = images.get_shape())
        y_ = tf.placeholder('float', shape = [None, 1])

        conv1W = tf.Variable(net_data["conv1"][0], name='conv1W', trainable=False)
        conv1b = tf.Variable(net_data["conv1"][1], name='conv1b', trainable=False)
        conv2W = tf.Variable(net_data["conv2"][0], name='conv2W', trainable=False)
        conv2b = tf.Variable(net_data["conv2"][1], name='conv2b', trainable=False)
        conv3W = tf.Variable(net_data["conv3"][0], name='conv3W')
        conv3b = tf.Variable(net_data["conv3"][1], name='conv3b')
        conv4W = tf.Variable(net_data["conv4"][0], name='conv4W')
        conv4b = tf.Variable(net_data["conv4"][1], name='conv4b')
        conv5W = tf.Variable(net_data["conv5"][0], name='conv5W')
        conv5b = tf.Variable(net_data["conv5"][1], name='conv5b')
       
        #conv1W = tf.Variable(tf.random_normal([11, 11, 3, 96]), 'conv1W')
        #conv1b = tf.Variable(tf.random_normal([96]), 'conv1b')
        #conv2W = tf.Variable(tf.random_normal([5, 5, 48, 256]), 'conv2W')
        #conv2b = tf.Variable(tf.random_normal([256]), 'conv2b')
        #conv3W = tf.Variable(tf.random_normal([3, 3, 256, 384]), 'conv3W')
        #conv3b = tf.Variable(tf.random_normal([384]), 'conv3b')
        #conv4W = tf.Variable(tf.random_normal([3, 3, 192, 384]), 'conv4W')
        #conv4b = tf.Variable(tf.random_normal([384]), 'conv4b')
        #conv5W = tf.Variable(tf.random_normal([3, 3, 192, 256]), 'conv5W')
        #conv5b = tf.Variable(tf.random_normal([256]), 'conv5b')

        fc6W = weight_variable([hidden_dim * 256, 4096], 'fc6W')
        fc6b = bias_variable([4096], 'fc6b')
        fc7W = weight_variable([4096, 4096], 'fc7W')
        fc7b = bias_variable([4096], 'fc7b')
        fc8W = weight_variable([4096, 1], 'fc8W')
        fc8b = bias_variable([1], 'fc8b')
        keep_prob = tf.placeholder('float')


        def model(x):
          # conv1
          conv1 = tf.nn.relu(conv(x, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1))

          lrn1 = tf.nn.local_response_normalization(conv1,
                              depth_radius=5,
                              alpha=0.0001,
                              beta=0.75,
                              bias=1.0)
          # maxpool1
          maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
          # conv2
          conv2 = tf.nn.relu(conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))

          lrn2 = tf.nn.local_response_normalization(conv2,
                              depth_radius=5,
                              alpha=0.0001,
                              beta=0.75,
                              bias=1.0)
          # maxpool2
          #maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
          # conv3
          conv3 = tf.nn.relu(conv(lrn2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))
          # conv4
          conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))
          # conv5
          conv5 = tf.nn.relu(conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))

         # print(int(conv5.get_shape()[0]), int(conv5.get_shape()[1]), int(conv5.get_shape()[2]))

          maxpool5 = spatial_pyramid_pool(conv5,
                          int(conv5.get_shape()[0]),
                           [int(conv5.get_shape()[1]), int(conv5.get_shape()[2])],
                           out_pool_size)
          # fc6

          fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
          fc6_drop = tf.nn.dropout(fc6, keep_prob)
          # fc7

          fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b)
          fc7_drop = tf.nn.dropout(fc7, keep_prob)

          fc8 = tf.nn.xw_plus_b(fc7_drop, fc8W, fc8b)
          return fc8
        
        logits = model(x)
        mean_squared = tf.reduce_mean(tf.square(tf.subtract(logits, y_)))
        regularizers = tf.nn.l2_loss(conv3W) + tf.nn.l2_loss(conv3b) + \
                 tf.nn.l2_loss(conv4W) + tf.nn.l2_loss(conv4b) + \
                 tf.nn.l2_loss(conv5W) + tf.nn.l2_loss(conv5b) + \
                 tf.nn.l2_loss(fc6W) + tf.nn.l2_loss(fc6b) + \
                 tf.nn.l2_loss(fc7W) + tf.nn.l2_loss(fc7b) + \
                 tf.nn.l2_loss(fc8W) + tf.nn.l2_loss(fc8b)

        loss = tf.reduce_mean(mean_squared + WEIGHT_DECAY * regularizers)

        # optimisation loss function
        global_step = tf.Variable(0)
        #learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.9, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(tf.losses.mean_squared_error(logits, y_), global_step = global_step)

        # evaluation
        rms = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, y_))))
        mean_abs = tf.reduce_mean(tf.abs(tf.subtract(logits,y_)))
        saver = tf.train.Saver({v.op.name: v for v in [conv1W, conv1b,
                                 conv2W, conv2b,
                                 conv3W, conv3b,
                                 conv4W, conv4b,
                                 conv5W, conv5b,
                                 fc6W, fc6b,
                                 fc7W, fc7b,
                                 fc8W, fc8b]})


      with tf.Session(graph=graph) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        if startedModel:
          saver.restore(sess, './my_model.ckpt')
        
        numFiles = metadata[train_f.split(".")[0]]
        counter = 0
        rms_sum = 0

        while counter < numFiles:
          xtrain, ytrain = sess.run([images, labels])
          _, curr_rms = sess.run([train_step, mean_abs], feed_dict = {x: xtrain, y_: ytrain, keep_prob: DROPOUT})
          rms_sum += curr_rms
          total_train_rms += curr_rms
          counter += BATCH_SIZE

        rms_sum /= math.ceil(float(numFiles) / BATCH_SIZE)
        total_train_count += math.ceil(float(numFiles) / BATCH_SIZE)
        #print("Training Abs Error for shape (" + w + ", " + h + "):", rms_sum)
        saver.save(sess, './my_model.ckpt')
        startedModel = True
        coord.request_stop()
        coord.join(threads)
   
      sess.close()
    
    total_train_rms /= total_train_count
    print("Training Absolute Error for epoch " + str(it) + ":", total_train_rms)
    total_val_rms = 0
    total_val_count = 0

    for val_f in val_files:

      split = val_f.split("_")
      w = split[1]
      h = split[2].split(".")[0]

      label = val_f.split(".")[0] + '/label' 
      image = val_f.split(".")[0] + '/image' 

      graph = tf.Graph()
      with graph.as_default():

        feature = {label: tf.FixedLenFeature([], tf.float32), image: tf.FixedLenFeature([], tf.string)}

        filename_queue = tf.train.string_input_producer([ABS_PATH + '/tf_files/' + val_f])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features[image], tf.float64)
        label = tf.cast(features[label], tf.float32)
        label = tf.reshape(label, [-1])
        image = tf.reshape(image, [int(w), int(h), 3])
        images, labels = tf.train.batch([image, label], \
                 batch_size=VAL_BATCH_SIZE, \
                 capacity=VAL_BATCH_SIZE, \
                 num_threads=8, \
                 allow_smaller_final_batch=False)

        x = tf.placeholder('float', shape = images.get_shape())
        y_ = tf.placeholder('float', shape = [None, 1])

        conv1W = tf.Variable(net_data["conv1"][0], name='conv1W')
        conv1b = tf.Variable(net_data["conv1"][1], name='conv1b')
        conv2W = tf.Variable(net_data["conv2"][0], name='conv2W')
        conv2b = tf.Variable(net_data["conv2"][1], name='conv2b')
        conv3W = tf.Variable(net_data["conv3"][0], name='conv3W')
        conv3b = tf.Variable(net_data["conv3"][1], name='conv3b')
        conv4W = tf.Variable(net_data["conv4"][0], name='conv4W')
        conv4b = tf.Variable(net_data["conv4"][1], name='conv4b')
        conv5W = tf.Variable(net_data["conv5"][0], name='conv5W')
        conv5b = tf.Variable(net_data["conv5"][1], name='conv5b')
        fc6W = weight_variable([hidden_dim * 256, 4096], 'fc6W')
        fc6b = bias_variable([4096], 'fc6b')
        fc7W = weight_variable([4096, 4096], 'fc7W')
        fc7b = bias_variable([4096], 'fc7b')
        fc8W = weight_variable([4096, 1], 'fc8W')
        fc8b = bias_variable([1], 'fc8b')
        keep_prob = tf.placeholder('float')


        def model(x):
          # conv1
          conv1 = tf.nn.relu(conv(x, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1))

          lrn1 = tf.nn.local_response_normalization(conv1,
                              depth_radius=5,
                              alpha=0.0001,
                              beta=0.75,
                              bias=1.0)
          # maxpool1
          maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
          # conv2
          conv2 = tf.nn.relu(conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))

          lrn2 = tf.nn.local_response_normalization(conv2,
                              depth_radius=5,
                              alpha=0.0001,
                              beta=0.75,
                              bias=1.0)
          # maxpool2
          #maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
          # conv3
          conv3 = tf.nn.relu(conv(lrn2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))
          # conv4
          conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))
          # conv5
          conv5 = tf.nn.relu(conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))

          maxpool5 = spatial_pyramid_pool(conv5,
                          int(conv5.get_shape()[0]),
                           [int(conv5.get_shape()[1]), int(conv5.get_shape()[2])],
                           out_pool_size)
          # fc6

          fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
          fc6_drop = tf.nn.dropout(fc6, keep_prob)
          # fc7

          fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b)
          fc7_drop = tf.nn.dropout(fc7, keep_prob)

          fc8 = tf.nn.xw_plus_b(fc7_drop, fc8W, fc8b)
          return fc8
        
        logits = model(x)

        # evaluation
        rms = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, y_))))
        mean_abs = tf.reduce_mean(tf.abs(tf.subtract(logits,y_)))
        saver = tf.train.Saver({v.op.name: v for v in [conv1W, conv1b,
                                 conv2W, conv2b,
                                 conv3W, conv3b,
                                 conv4W, conv4b,
                                 conv5W, conv5b,
                                 fc6W, fc6b,
                                 fc7W, fc7b,
                                 fc8W, fc8b]})


      with tf.Session(graph=graph) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if startedModel:
          saver.restore(sess, './my_model.ckpt')

        numFiles = metadata[val_f.split(".")[0]]
        counter = 0
        rms_sum = 0

        while counter < numFiles:

          xtrain, ytrain = sess.run([images, labels])
          curr_rms = sess.run(mean_abs, 
                          feed_dict = {x: xtrain,
                                 y_: ytrain, 
                                 keep_prob: 1.0})

          rms_sum += curr_rms
          total_val_rms += curr_rms
          counter += VAL_BATCH_SIZE


        rms_sum /= math.ceil(float(numFiles) / VAL_BATCH_SIZE)
        total_val_count += math.ceil(float(numFiles) / VAL_BATCH_SIZE)
        #print("Validation RMS for shape (" + w + ", " + h + "):", rms_sum)

        coord.request_stop()
        coord.join(threads)

      sess.close()

    total_val_rms /= total_val_count
    print("Validation Mean Absolute Error for epoch " + str(it) + ":", total_val_rms, "\n")
    sys.stdout.flush()
    if total_val_rms < best_val:
      best_val = total_val_rms
      best_d = DROPOUT
      best_w = WEIGHT_DECAY
      best_l = LEARNING_RATE

print("**Best Hyperparameters**\n")
print("Learning rate:", best_l)
print("Dropout:", best_d)
print("Weight decay:", best_w)
sys.stdout.flush()
