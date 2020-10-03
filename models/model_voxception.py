# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Hao Zhu, Zhan Ma, Tong Chen, Haojie Liu, Qiu Shen; Nanjing University, Vision Lab.
# Chaofei Wang; Shanghai Jiao Tong University, Cooperative Medianet Innovation Center.
# Last update: 2019.06.14 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class _VoxceptionResNet(tf.keras.Model):
  """Voxception Residual Network Block.

  Arguments:
    num_filters: number of filters passed to a convolutional layer.
  """

  def __init__(self, num_filters, name, activation=tf.nn.relu):
    super(_VoxceptionResNet, self).__init__(name='')
    #path_1
    self.conv1_1 = tf.keras.layers.Conv3D(int(num_filters/4), 
                                          (3,3,3), 
                                          padding='same', 
                                          activation=activation,
                                          use_bias=True, 
                                          name=name+'_conv1_1')

    self.conv1_2 = tf.keras.layers.Conv3D(int(num_filters/2), 
                                          (3,3,3), 
                                          padding='same', 
                                          activation=activation, 
                                          use_bias=True, 
                                          name=name+'_conv1_2')
    #path_2
    self.conv2_1 = tf.keras.layers.Conv3D(int(num_filters/4), 
                                          (1,1,1), 
                                          padding='same',  
                                          activation=activation,
                                          use_bias=True, 
                                          name=name+'_conv2_1')

    self.conv2_2 = tf.keras.layers.Conv3D(int(num_filters/4), 
                                          (3,3,3), 
                                          padding='same',  
                                          activation=activation,
                                          use_bias=True, 
                                          name=name+'_conv2_2') 

    self.conv2_3 = tf.keras.layers.Conv3D(int(num_filters/2), 
                                          (1,1,1), 
                                          padding='same',  
                                          activation=activation,
                                          use_bias=True, 
                                          name=name+'_conv2_3')

  def call(self, x):
    # path1
    tensor1_1 = self.conv1_1(x)
    tensor1_2 = self.conv1_2(tensor1_1)
    # path2
    tensor2_1 = self.conv2_1(x)
    tensor2_2 = self.conv2_2(tensor2_1)
    tensor2_3 = self.conv2_3(tensor2_2)
    # concat paths
    residual = tf.concat([tensor1_2, tensor2_3], axis=-1)
    # add & relu
    output = tf.nn.relu(x + residual)
    return output


class AnalysisTransform(tf.keras.Model):
  """Analysis transformation.

  Arguments:
    None.
  """
  def __init__(self):
    super(AnalysisTransform, self).__init__()

    def vrn_block(num_filters, name):
      return _VoxceptionResNet(num_filters, name)
    
    self.conv_in = tf.keras.layers.Conv3D(16, 
                                          (3,3,3), 
                                          padding='same', 
                                          activation=tf.nn.relu, 
                                          use_bias=True, 
                                          name="conv_in")
    
    self.vrn1_1 = vrn_block(16, 'vrn1_1')
    self.vrn1_2 = vrn_block(16, 'vrn1_2')
    self.vrn1_3 = vrn_block(16, 'vrn1_3')
    
    self.down_1 = tf.keras.layers.Conv3D(32, 
                                        (3,3,3), 
                                        (2,2,2), 
                                        padding='same', 
                                        activation=tf.nn.relu, 
                                        use_bias=False, 
                                        name="down_1")
    
    self.vrn2_1 = vrn_block(32, 'vrn2_1')
    self.vrn2_2 = vrn_block(32, 'vrn2_2')
    self.vrn2_3 = vrn_block(32, 'vrn2_3')
    
    self.down_2 = tf.keras.layers.Conv3D(64, 
                                        (3,3,3), 
                                        (2,2,2), 
                                        padding='same', 
                                        activation=tf.nn.relu, 
                                        use_bias=False, 
                                        name="down_2")
    
    self.vrn3_1 = vrn_block(64, 'vrn3_1')
    self.vrn3_2 = vrn_block(64, 'vrn3_2')
    self.vrn3_3 = vrn_block(64, 'vrn3_3')   
    
    self.conv_out = tf.keras.layers.Conv3D(16, 
                                          (3,3,3), 
                                          padding='same', 
                                          use_bias=True, 
                                          name="conv_out")
  
  # @tf.contrib.eager.defun
  def call(self, x):
    
    feature1 = self.conv_in(x)# [N,N,N,16]
    feature1_1 = self.vrn1_1(feature1)
    feature1_2 = self.vrn1_2(feature1_1)
    feature1_3 = self.vrn1_3(feature1_2)# [N,N,N,16]
    
    feature2 = self.down_1(feature1_3)# [N/2,N/2,N/2,32]
    feature2_1 = self.vrn2_1(feature2)
    feature2_2 = self.vrn2_2(feature2_1)
    feature2_3 = self.vrn2_3(feature2_2)# [N/2,N/2,N/2,32]
    
    feature3 = self.down_2(feature2_3)# [N/4,N/4,N/4,64]
    feature3_1 = self.vrn3_1(feature3)
    feature3_2 = self.vrn3_2(feature3_1)
    feature3_3 = self.vrn3_3(feature3_2)# [N/4,N/4,N/4,64]
    
    feature4 = self.conv_out(feature3_3)# [N/4,N/4,N/4,16]

    return feature4


class SynthesisTransform(tf.keras.Model):
  def __init__(self):
    super(SynthesisTransform, self).__init__()
    def vrn_block(num_filters, name):
      return _VoxceptionResNet(num_filters, name)
    
    self.deconv_in = tf.keras.layers.Conv3D(64, 
                                            (3,3,3), 
                                            padding='same', 
                                            activation=tf.nn.relu, 
                                            use_bias=True, 
                                            name="deconv_in")
    
    self.vrn1_1 = vrn_block(64, 'dvrn1_1')
    self.vrn1_2 = vrn_block(64, 'dvrn1_2')
    self.vrn1_3 = vrn_block(64, 'dvrn1_3')
    
    self.up_1 = tf.keras.layers.Conv3DTranspose(32, 
                                                (3,3,3), 
                                                (2,2,2), 
                                                padding='same', 
                                                activation=tf.nn.relu, 
                                                use_bias=True, 
                                                name="up_1")
    
    self.vrn2_1 = vrn_block(32, 'dvrn2_1')
    self.vrn2_2 = vrn_block(32, 'dvrn2_2')
    self.vrn2_3 = vrn_block(32, 'dvrn2_3')
    
    self.up_2 = tf.keras.layers.Conv3DTranspose(16, 
                                                (3,3,3), 
                                                (2,2,2), 
                                                padding='same', 
                                                activation=tf.nn.relu, 
                                                use_bias=True, 
                                                name="up_2")
    
    self.vrn3_1 = vrn_block(16, 'dvrn3_1')
    self.vrn3_2 = vrn_block(16, 'dvrn3_2')
    self.vrn3_3 = vrn_block(16, 'dvrn3_3') 
    # 
    self.deconv_out = tf.keras.layers.Conv3D(1, 
                                            (3,3,3), 
                                            padding='same', 
                                            use_bias=True, 
                                            name="deconv_out")
  
  # @tf.contrib.eager.defun
  def call(self, x):
    
    feature1 = self.deconv_in(x)# [N/4,N/4,N/4,64]
    feature1_1 = self.vrn1_1(feature1)
    feature1_2 = self.vrn1_2(feature1_1)
    feature1_3 = self.vrn1_3(feature1_2)# [N/4,N/4,N/4,64]
    
    feature2 = self.up_1(feature1_3)# [N/2,N/2,N/2,32]
    feature2_1 = self.vrn2_1(feature2)
    feature2_2 = self.vrn2_2(feature2_1)
    feature2_3 = self.vrn2_3(feature2_2)# [N/2,N/2,N/2,32]
    
    feature3 = self.up_2(feature2_3)# [N,N,N,16]
    feature3_1 = self.vrn3_1(feature3)
    feature3_2 = self.vrn3_2(feature3_1)
    feature3_3 = self.vrn3_3(feature3_2)# [N,N,N,16]
    
    feature4 = self.deconv_out(feature3_3)# [N,N,N,1]

    return feature4


class HyperEncoder(tf.keras.Model):
  """Hyper encoder.
  """

  def __init__(self, activation=tf.nn.relu):
    super(HyperEncoder, self).__init__()

    self.conv1 = tf.keras.layers.Conv3D(16, 
                                        (3,3,3), 
                                        padding='same', 
                                        activation=activation,
                                        use_bias=True, 
                                        name='conv1')
      
    self.conv2 = tf.keras.layers.Conv3D(16, 
                                        (3,3,3), 
                                        (2,2,2),
                                        padding='same', 
                                        activation=activation,
                                        use_bias=True, 
                                        name='conv2')

    self.conv3 = tf.keras.layers.Conv3D(8, 
                                        (3,3,3), 
                                        padding='same', 
                                        activation=None,
                                        use_bias=True, 
                                        name='conv3')

  def call(self, x):

    f1 = self.conv1(x)
    f2 = self.conv2(f1)
    f3 = self.conv3(f2)

    return f3


class HyperDecoder(tf.keras.Model):
  """Hyper decoder.
    Return: location, scale.
  """

  def __init__(self, activation=tf.nn.relu):
    super(HyperDecoder, self).__init__()

    self.conv1 = tf.keras.layers.Conv3D(16, 
                                        (3,3,3), 
                                        padding='same', 
                                        activation=activation,
                                        use_bias=True, 
                                        name='deconv1')
      
    self.conv2 = tf.keras.layers.Conv3DTranspose(16, 
                                                (3,3,3), 
                                                (2,2,2),
                                                padding='same', 
                                                activation=activation,
                                                use_bias=True, 
                                                name='deconv2')

    self.conv3 = tf.keras.layers.Conv3D(32, 
                                        (3,3,3), 
                                        padding='same', 
                                        activation=activation,
                                        use_bias=True, 
                                        name='deconv3')

    self.conv4_1 = tf.keras.layers.Conv3D(16, 
                                          (3,3,3), 
                                          padding='same', 
                                          activation=None,
                                          use_bias=True, 
                                          name='deconv4_1')

    self.conv4_2 = tf.keras.layers.Conv3D(16, 
                                          (3,3,3), 
                                          padding='same', 
                                          activation=None,
                                          use_bias=True, 
                                          name='deconv4_2')

  def call(self, x):

    f1 = self.conv1(x)
    f2 = self.conv2(f1)
    f3 = self.conv3(f2)

    loc = self.conv4_1(f3)
    scale = self.conv4_2(f3)

    return loc, tf.abs(scale)

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

    # hyper_encoder & hyper_decoder
    hyper_encoder = HyperEncoder()
    hyper_decoder = HyperDecoder()
    #
    hyper_prior = hyper_encoder(features)
    print(hyper_prior)
    loc, scale = hyper_decoder(hyper_prior)
    print(loc, scale)

    print(encoder.summary())
    print(decoder.summary())

    print(hyper_encoder.summary())
    print(hyper_decoder.summary())


