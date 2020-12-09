# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2020.12.1 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class AnalysisTransform(tf.keras.Model):
  """Analysis transformation.

  Arguments:
    None.
  """
  def __init__(self):
    super(AnalysisTransform, self).__init__()
    
    self.conv_1 = tf.keras.layers.Conv3D(32, 
                                        (9,9,9), 
                                        (2,2,2), 
                                        padding='same', 
                                        activation=tf.nn.relu, 
                                        use_bias=True, 
                                        name="conv_1")
    
    self.conv_2 = tf.keras.layers.Conv3D(32, 
                                        (5,5,5), 
                                        (2,2,2), 
                                        padding='same', 
                                        activation=tf.nn.relu, 
                                        use_bias=True, 
                                        name="conv_2")
    
    self.conv_3 = tf.keras.layers.Conv3D(32, 
                                        (5,5,5), 
                                        (2,2,2), 
                                        padding='same', 
                                        use_bias=False, 
                                        name="conv_3")
  
  # @tf.contrib.eager.defun
  def call(self, x):
    
    feature1 = self.conv_1(x)# 
    feature2 = self.conv_2(feature1)# 
    feature3 = self.conv_3(feature2)# 

    return feature3


class SynthesisTransform(tf.keras.Model):
  def __init__(self):
    super(SynthesisTransform, self).__init__()
    
    self.deconv_1 = tf.keras.layers.Conv3DTranspose(32, 
                                                    (5,5,5), 
                                                    (2,2,2), 
                                                    padding='same', 
                                                    activation=tf.nn.relu, 
                                                    use_bias=True, 
                                                    name="deconv_1")
    
    self.deconv_2 = tf.keras.layers.Conv3DTranspose(32, 
                                                    (5,5,5), 
                                                    (2,2,2), 
                                                    padding='same', 
                                                    activation=tf.nn.relu, 
                                                    use_bias=True, 
                                                    name="deconv_2")

    # self.deconv_3 = tf.keras.layers.Conv3DTranspose(1, 
    #                                                 (9,9,9), 
    #                                                 (2,2,2), 
    #                                                 padding='same', 
    #                                                 activation=tf.nn.relu, 
    #                                                 use_bias=True, 
    #                                                 name="deconv_3")
    self.deconv_3 = tf.keras.layers.Conv3DTranspose(1, 
                                                    (9,9,9), 
                                                    (2,2,2), 
                                                    padding='same', 
                                                    use_bias=True, 
                                                    name="deconv_3")

  # @tf.contrib.eager.defun
  def call(self, x):
    
    feature1 = self.deconv_1(x)# 
    feature2 = self.deconv_2(feature1)# 
    feature3 = self.deconv_3(feature2)# 

    return feature3


if __name__=='__main__':
  with tf.Graph().as_default():
    # inputs
    inputs = tf.cast(tf.zeros((4,64,64,64,1)), "float32")
    # encoder & decoder
    encoder = AnalysisTransform()
    decoder = SynthesisTransform()
    #
    features = encoder(inputs)
    print(features)
    outputs = decoder(features)
    print(outputs)

    print(encoder.summary())
    print(decoder.summary())

