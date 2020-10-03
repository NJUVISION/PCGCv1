# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Hao Zhu, Zhan Ma, Tong Chen, Haojie Liu, Qiu Shen; Nanjing University, Vision Lab.
# Chaofei Wang; Shanghai Jiao Tong University, Cooperative Medianet Innovation Center.
# Last update: 2019.10.07

import tensorflow as tf 
import numpy as np
from tensorflow.contrib.coder.python.ops import coder_ops

class SymmetricConditional(tf.keras.layers.Layer):
  """Symmetric conditional entropy model.

  Argument:
    likelihood_bound;
    range_coder_precision;
  """

  def __init__(self, likelihood_bound=1e-9, range_coder_precision=16):
    super(SymmetricConditional, self).__init__()
    self._likelihood_bound = float(likelihood_bound)
    self._range_coder_precision = int(range_coder_precision)

  def _standardized_cumulative(self, inputs, loc, scale):
    """
    Laplace cumulative densities function.
    """

    mask_r = tf.math.greater(inputs,loc)
    mask_l = tf.math.less_equal(inputs,loc)
    c_l = 1.0/2.0 * tf.math.exp(-tf.abs(inputs - loc) / scale)
    c_r = 1.0 - 1.0/2.0 * tf.exp(-tf.abs(inputs - loc) / scale)
    c = c_l*tf.cast(mask_l,dtype=tf.float32) + c_r*tf.cast(mask_r,dtype=tf.float32)
    
    return c

  def _likelihood(self, inputs, loc, scale):
    """ Estimate the likelihoods conditioned on assumed distribution.

    Arguments:
      inputs;(quantized values); loc; scale;
    Return:
      likelihood.
    """

    # CDF.
    upper = inputs + .5
    lower = inputs - .5
    sign = tf.math.sign(upper + lower - loc)
    upper = - sign * (upper - loc) + loc
    lower = - sign * (lower - loc) + loc

    cdf_upper = self._standardized_cumulative(upper, loc, scale)
    cdf_lower = self._standardized_cumulative(lower, loc, scale)

    likelihood = tf.math.abs(cdf_upper - cdf_lower)

    return likelihood

  def _quantize(self, inputs, mode):
    """Add noise or quantize."""
    half = tf.constant(.5, dtype=self.dtype)

    if mode == "noise":
      noise = tf.random.uniform(tf.shape(inputs), -half, half)
      return tf.math.add_n([inputs, noise])

    if mode == "symbols":
      outputs = tf.math.round(inputs)
      # outputs = tf.cast(outputs, tf.int32)
      return outputs

  def call(self, inputs, loc, scale, training):
    """Pass a tensor through the bottleneck.

    Arguments:
      input tensor, loc, scale.

    Returns:
      output quantized tensor.
      likelihoods.
    """

    # check.
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    loc = tf.convert_to_tensor(loc, dtype=self.dtype)
    scale = tf.convert_to_tensor(scale, dtype=self.dtype)
    # quantize.
    outputs = self._quantize(inputs, "noise" if training else "symbols")
    # likelihood.
    likelihood = self._likelihood(outputs, loc, scale)
    likelihood_bound = tf.constant(self._likelihood_bound, dtype=self.dtype)
    likelihood = tf.maximum(likelihood, likelihood_bound)

    return outputs, likelihood

  def _get_cdf(self, loc, scale, min_v, max_v, datashape):
    """Get quantized cdf for compress/decompress.
    
    Arguments:
      inputs: integer tensor min_v, max_v. 
              float32 tensor loc, scale. [-1, channels]
    Return: 
      cdf with shape [-1, channels, symbols]
    """

    # shape of cdf shound be # [-1, C, N]
    a = tf.reshape(tf.range(min_v, max_v+1), [1, 1, max_v-min_v+1])
    channels = datashape[-1]
    a = tf.tile(a, [tf.reduce_prod(datashape)/channels, channels, 1])
    a = tf.cast(a, tf.float32)# [-1, C, N]

    loc = tf.expand_dims(loc, -1)
    scale = tf.expand_dims(scale, -1)
    likelihood = self._likelihood(a, loc, scale)
    likelihood_bound = tf.constant(self._likelihood_bound, dtype=self.dtype)
    likelihood = tf.maximum(likelihood, likelihood_bound)
    pmf = likelihood

    cdf = coder_ops.pmf_to_quantized_cdf(pmf, precision=self._range_coder_precision)# [-1, C, N]

    return cdf

  def compress(self, inputs, loc, scale):
    """Compress inputs and store their binary representations into strings.

    Arguments:
      inputs: `Tensor` with values to be compressed. Must have shape 
      [**batch size**, length, width, height, channels]
      locs & scales: same shape like inputs.
    Returns:
      compressed: String `Tensor` vector containing the compressed
        representation of each batch element of `inputs`.
    """

    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    loc = tf.convert_to_tensor(loc, dtype=self.dtype)
    scale = tf.convert_to_tensor(scale, dtype=self.dtype)

    datashape = tf.shape(inputs)
    channels = datashape[-1]

    # reshape.
    loc = tf.reshape(loc, [-1, channels])
    scale = tf.reshape(scale, [-1, channels])
    inputs = tf.reshape(inputs, [-1, channels])

    # quantize.
    values = self._quantize(inputs, "symbols")
    # get cdf
    min_v = tf.cast(tf.floor(tf.reduce_min(values)), dtype=tf.int32)
    max_v = tf.cast(tf.ceil(tf.reduce_max(values)), dtype=tf.int32)
    cdf = self._get_cdf(loc, scale, min_v, max_v, datashape)# [BatchSizexHxWxD, C, N]

    # range encode.
    values = tf.cast(values, "int32")
    values -= min_v
    values = tf.cast(values, "int16")
    strings = coder_ops.range_encode(values, cdf, precision=self._range_coder_precision)

    return strings, min_v, max_v

  def decompress(self, strings, loc, scale, min_v, max_v, datashape):
    """Decompress values from their compressed string representations.

    Arguments:
      strings: A string `Tensor` vector containing the compressed data.
      shape: A `Tensor` vector of int32 type. Contains the shape of the tensor to be
        decompressed. [batch size, length, width, height, channels]
      loc & scale: parameters of distributions.
      min_v & max_v: minimum & maximum values.
    
    Return: outputs [BatchSize, H, W, D, C]
    """

    strings = tf.convert_to_tensor(strings, dtype=tf.string)
    datashape = tf.convert_to_tensor(datashape, dtype='int32')
    loc = tf.convert_to_tensor(loc, dtype=self.dtype)
    scale = tf.convert_to_tensor(scale, dtype=self.dtype)
    min_v = tf.convert_to_tensor(min_v, dtype='int32')
    max_v = tf.convert_to_tensor(max_v, dtype='int32')
  
    # reshape.
    channels = datashape[-1]
    loc = tf.reshape(loc, [-1, channels])
    scale = tf.reshape(scale, [-1, channels])

    # get cdf.
    cdf = self._get_cdf(loc, scale, min_v, max_v, datashape)# [BatchSizexHxWxD, C, N]

    # range decode.
    code_shape = (tf.reduce_prod(datashape)/channels, channels)# shape=[-1, channel]
    values = coder_ops.range_decode(strings, code_shape, cdf, precision=self._range_coder_precision)
    values = tf.cast(values, tf.int32)
    values = values + min_v
    values = tf.reshape(values, datashape)
    values = tf.cast(values, "float32")

    return values



    


