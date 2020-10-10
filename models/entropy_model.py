# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2019.09.29

import tensorflow as tf 
import numpy as np
from tensorflow.contrib.coder.python.ops import coder_ops

class EntropyBottleneck(tf.keras.layers.Layer):
  """The layer implements a flexible probability density model to estimate
  entropy of its input tensor, which is described in this paper:
  >"Variational image compression with a scale hyperprior"
  > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
  > https://arxiv.org/abs/1802.01436
  """

  def __init__(self, likelihood_bound=1e-9, range_coder_precision=16, 
                init_scale=8, filters=(3,3,3)):
    super(EntropyBottleneck, self).__init__()
    self._likelihood_bound = float(likelihood_bound)
    self._range_coder_precision = int(range_coder_precision)
    self._init_scale = float(init_scale)
    self._filters = tuple(int(f) for f in filters)
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape):
    """Build the entropy model.
    
    Creates the variables for the network modeling the densities.
    And then uses that to create the probability mass functions (pmf) and the
    discrete cumulative density functions (cdf) used by the range coder.

    Arguments:
      input_shape. 
    """

    input_shape = tf.TensorShape(input_shape)
    channel_axes = input_shape.ndims - 1# channel last.
    channels = input_shape[channel_axes].value
    self.input_spec = tf.keras.layers.InputSpec(
      ndim=input_shape.ndims, axes={channel_axes: channels})

    filters = (1,) + self._filters +(1,)
    scale = self._init_scale ** (1 / (len(self._filters) + 1))

    # Create variables.
    self._matrices = []
    self._biases = []
    self._factors = []
    for i in range(len(self._filters) + 1):
      init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
      matrix = self.add_variable(
        "matrix_{}".format(i), dtype=self.dtype,
        shape=(channels, filters[i + 1], filters[i]),
        initializer=tf.initializers.constant(init))
      self._matrices.append(matrix)

      bias = self.add_variable(
        "bais_{}".format(i), dtype=self.dtype,
        shape=(channels, filters[i+1], 1),
        initializer=tf.initializers.random_uniform(-.5, .5))
      self._biases.append(bias)

      factor = self.add_variable(
        "factor_{}".format(i), dtype=self.dtype,
        shape=(channels, filters[i + 1], 1),
        initializer=tf.initializers.zeros()) 
      # factor = tf.math.tanh(factor)# !
      self._factors.append(factor)

    super(EntropyBottleneck, self).build(input_shape)

  def _logits_cumulative(self, inputs):
    """Evaluate logits of the cumulative densities.
    
    Arguments:
      inputs: The values at which to evaluate the cumulative densities,
        expected to have shape `(channels, 1, batch)`.

    Returns:
      A tensor of the same shape as inputs, containing the logits of the
      cumulatice densities evaluated at the the given inputs.
      """

    logits = inputs

    for i in range(len(self._filters) + 1):
      matrix = self._matrices[i]
      matrix = tf.nn.softplus(matrix)
      logits = tf.linalg.matmul(matrix, logits)

      bias = self._biases[i]
      logits += bias
      
      factor = self._factors[i]
      factor = tf.math.tanh(factor)
      logits = logits + factor * tf.math.tanh(logits)

    return logits

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

  def _likelihood(self, inputs):
    """ Estimate the likelihoods.

    Arguments:
      inputs: tensor with shape(batch size, length, width, height, channels) 

    Return:
      likelihoods: tensor with shape(batch size, length, width, height, channels) 
    """

    ndim = self.input_spec.ndim
    channel_axes = ndim - 1 
    half = tf.constant(.5, dtype=self.dtype)

    # Convert shape to (channels, 1, -1)
    order = list(range(ndim))# order=[0,1,2,3,4]
    order.pop(channel_axes)# order=[0,1,2,3] 
    order.insert(0, channel_axes)# order=[4,0,1,2,3]
    inputs = tf.transpose(inputs, order)
    shape = tf.shape(inputs)# shape=[channels, batch, length, width, height]
    inputs = tf.reshape(inputs, (shape[0], 1, -1))# shape=(channel, 1, -1)

    # Evaluate densities.
    lower = self._logits_cumulative(inputs - half)
    upper = self._logits_cumulative(inputs + half)

    # Flip signs if we can move more towards the left tail of the sigmoid.
    sign = -tf.math.sign(tf.math.add_n([lower, upper]))
    # sign = tf.stop_gradient(sign)
    likelihood = tf.math.abs(tf.math.sigmoid(sign * upper) - tf.math.sigmoid(sign * lower))
    
    # Convert back to input tensor shape.
    order = list(range(1, ndim))# order=[1,2,3,4]
    order.insert(channel_axes, 0)# order=[1,2,3,4,0]
    likelihood = tf.reshape(likelihood, shape)# shape=[channels, batch, length, width, height]
    likelihood = tf.transpose(likelihood, order)# shape=[batch size, length, width, height, channels]

    return likelihood

  def call(self, inputs, training):
    """Pass a tensor through the bottleneck.
    
    Arguments:
      inputs: The tensor to be passed through the bottleneck.
      
      Returns:
        values: `Tensor` with the shape as `inputs` containing the perturbed
        or quantized input values.
        likelihood: `Tensor` with the same shape as `inputs` containing the
        likelihood of `values` under the modeled probability distributions.
    """

    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    outputs = self._quantize(inputs, "noise" if training else "symbols")

    # Evaluate densities.
    likelihood = self._likelihood(outputs)
    likelihood_bound = tf.constant(self._likelihood_bound, dtype=self.dtype)
    likelihood = tf.maximum(likelihood, likelihood_bound)

    # # TODO:delete graph execution.
    # if not tf.executing_eagerly():
    #   outputs_shape, likelihood_shape = \
    #   tf.TensorShape(inputs.shape), tf.TensorShape(inputs.shape)
    #   outputs.set_shape(outputs_shape)
    #   likelihood.set_shape(likelihood_shape)

    return outputs, likelihood

  def _get_cdf(self, min_v, max_v):
    """Get quantized cumulative density function (CDF) for compress/decompress.
    
    Arguments:
      inputs: integer tesnor min_v, max_v.
    Return: 
      cdf with shape [1, channels, symbols].
    """

    # TODO 
    # if there is only one symbol like 0, you can not convert pmf to quantized cdf!

    # get channels.
    ndim = self.input_spec.ndim
    channel_axes = ndim - 1
    channels = self.input_spec.axes[channel_axes]

    # shape of cdf shound be [C, 1, N]
    a = tf.reshape(tf.range(min_v, max_v+1), [1, 1, max_v-min_v+1])# [1, 1, N]
    a = tf.tile(a, [channels, 1, 1])# [C, 1, N]
    a = tf.cast(a, tf.float32)

    # estimate pmf/likelihood.
    half = tf.constant(.5, dtype=self.dtype)
    lower = self._logits_cumulative(a - half)
    upper = self._logits_cumulative(a + half)

    sign = -tf.math.sign(tf.math.add_n([lower, upper]))
    
    likelihood = tf.math.abs(tf.math.sigmoid(sign * upper) - tf.math.sigmoid(sign * lower))
    likelihood_bound = tf.constant(self._likelihood_bound, dtype=self.dtype)
    likelihood = tf.maximum(likelihood, likelihood_bound)
    pmf = likelihood
    
    # pmf to cdf.
    cdf = coder_ops.pmf_to_quantized_cdf(pmf, precision=self._range_coder_precision)
    cdf = tf.reshape(cdf, [1, channels, -1])

    return cdf

  def compress(self, inputs):
    """Compress inputs and store their binary representations into strings.

    Arguments:
      inputs: `Tensor` with values to be compressed. Must have shape 
      [**batch size**, length, width, height, channels]
    Returns:
      compressed: String `Tensor` vector containing the compressed
        representation of each batch element of `inputs`.
    """

    with tf.name_scope(self._name_scope()):
      inputs = tf.convert_to_tensor(inputs)
      if not self.built:
        if self.dtype is None:
          self._dtype = inputs.dtype.base_dtype.name
        self.build(inputs.shape)

      ndim = self.input_spec.ndim
      channel_axes = ndim - 1
      channels = self.input_spec.axes[channel_axes]

      # quantize.
      values = self._quantize(inputs, "symbols")

      # get cdf
      min_v = tf.cast(tf.floor(tf.reduce_min(values)), dtype=tf.int32)
      max_v = tf.cast(tf.ceil(tf.reduce_max(values)), dtype=tf.int32)
      cdf = self._get_cdf(min_v, max_v)

      # range encode.
      values = tf.reshape(values, [-1, channels])
      values = tf.cast(values, tf.int32)
      values = values - min_v
      values = tf.cast(values, tf.int16)
      strings = coder_ops.range_encode(
        values, cdf, precision=self._range_coder_precision)
      
      return strings, min_v, max_v

  def decompress(self, strings, min_v, max_v, shape, channels=None):
    """Decompress values from their compressed string representations.

    Arguments:
      strings: A string `Tensor` vector containing the compressed data.
      shape: A `Tensor` vector of int32 type. Contains the shape of the tensor to be
        decompressed. [batch size, length, width, height, channels]
      min_v & max_v: minimum & maximum values.
      
    Returns:
      The decompressed `Tensor`. tf.float32.
    """

    with tf.name_scope(self._name_scope()):
      strings = tf.convert_to_tensor(strings, dtype=tf.string)
      shape = tf.convert_to_tensor(shape, dtype='int32')# [batch size, length, width, height, channels]
      min_v = tf.convert_to_tensor(min_v, dtype='int32')
      max_v = tf.convert_to_tensor(max_v, dtype='int32')
      if self.built:
        ndim = self.input_spec.ndim
        if channels is None:
          channels = self.input_spec.axes[ndim - 1]
      else:
        channels = int(channels)
        ndim = shape.shape[0].value # ndim=5
        input_shape = ndim * [None]
        input_shape[-1] = channels
        self.build(input_shape)
      code_length = tf.reduce_prod(shape)
      code_shape = (tf.constant(code_length//channels, dtype='int32'), tf.constant(channels, dtype='int32'))# shape=[-1, channel]

      # get cdf.
      cdf = self._get_cdf(min_v, max_v)
      
      # range decode.
      values = coder_ops.range_decode(
          strings, code_shape, cdf, precision=self._range_coder_precision)
      values = tf.cast(values, tf.int32)
      values = values + min_v
      
      values = tf.reshape(values, shape)
      values = tf.cast(values, tf.float32)
      
      return values

