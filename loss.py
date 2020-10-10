# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2019.10.05

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

def get_bce_loss(pred, label):
  """ (Weighted) Binary cross entropy loss.
  Input:
      pred: [batch size, vsize, vsize, vsize, 1] float32
      label: must be 0 or 1, [batch size, vsize, vsize, vsize, 1] float32
  output: 
      empty loss & full loss
  """

  # occupancy = pred
  occupancy = tf.clip_by_value(tf.sigmoid(pred), 1e-7, 1.0 - 1e-7)
  # 1. location loss
  # get position from label
  position_neg = tf.cast(tf.equal(tf.reduce_max(label, axis=-1), 0), 'int8')
  position_pos = tf.cast(tf.greater(tf.reduce_max(label, axis=-1), 0), 'int8')
  # get position of pred
  position_neg = tf.where(position_neg>0)
  position_pos = tf.where(position_pos>0)
  # get occupancy value
  occupancy_neg = tf.gather_nd(occupancy, position_neg) 
  occupancy_pos = tf.gather_nd(occupancy, position_pos) 
  # get empty loss
  empty_loss = tf.reduce_mean(tf.negative(tf.log(1.0 - occupancy_neg)))
  full_loss = tf.reduce_mean(tf.negative(tf.log(occupancy_pos)))  

  return empty_loss, full_loss

def get_confusion_matrix(pred, label, th=0.):
  """confusion matrix: 
      1   0
    1 TP  FN
    0 FP  TN(option)
  input:
    pred, label: float32 [batch size, vsize, vsize, vsize, 1]
  output: 
    TP(true position), FP(false position), FN(false negative);
    float32 [batch size, vsize, vsize, vsize];
  """

  pred = tf.squeeze(pred, -1)
  label = tf.squeeze(label, -1)

  pred = tf.cast(tf.greater(pred, th), tf.float32)
  label = tf.cast(tf.greater(label, th), tf.float32)

  TP = pred * label
  FP = pred * (1. - label)
  FN = (1. - pred) * label
  # TN = (1 - pred) * (1 - label)

  return TP, FP, FN

def get_classify_metrics(pred, label, th=0.):
  """Metrics for classification.
  input:
      pred, label; type : float32 tensor;  shape: [batch size, vsize, vsize, vsize, 1]
  output:
      precision rate; recall rate; IoU;
  """

  TP, FP, FN = get_confusion_matrix(pred, label, th=th)

  TP = tf.cast(tf.reduce_sum(TP), tf.float32)
  FP = tf.cast(tf.reduce_sum(FP), tf.float32)
  FN = tf.cast(tf.reduce_sum(FN), tf.float32)

  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  IoU = TP / (TP + FP + FN)

  return precision, recall, IoU


if __name__=='__main__':
  np.random.seed(108)
  data = np.random.rand(2, 64, 64, 64, 1)* 10 - 5
  data = data.astype("float32")
  label = np.random.rand(2, 64, 64, 64, 1)
  label[label>=0.97] = 1
  label[label<0.97] = 0
  label = label.astype("float32")

  data = tf.Variable(data)
  label = tf.constant(label)

  if tf.executing_eagerly():
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
    for i in range(1000):
      with tf.GradientTape() as tape:
        loss1, loss2 = get_bce_loss(data, label)
        loss = loss1 + 3*loss2

        gradients = tape.gradient(loss, data)
        optimizer.apply_gradients([(gradients, data)])

        if i%100==0:
          print(i,loss.numpy())
          precision, recall, IoU = get_classify_metrics(data, label, 0.)
          print(precision.numpy(), recall.numpy(), IoU.numpy())

  if not tf.executing_eagerly():
    with tf.Session('') as sess:
      loss1, loss2 = get_bce_loss(data, label)
      loss = loss1 + 3*loss2
      train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
      sess.run(tf.global_variables_initializer())

      for i in range(1000):
        trainloss, _ = sess.run([loss,train])
        if i%100==0:
          print(i,trainloss)

          precision, recall, IoU = get_classify_metrics(data, label, 0.)
          print(sess.run([precision, recall, IoU]))

