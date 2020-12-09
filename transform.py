# Copyright (c) Nanjing University, Vision Lab.
# Last update:
# 2020.11.26 
# 2019.11.13 
# 2019.10.27
# 2019.10.07
# 2019.10.08

import os
import argparse
import numpy as np
import tensorflow as tf
import time
import importlib 
import subprocess
tf.enable_eager_execution()

import models.model_voxception as model
from models.entropy_model import EntropyBottleneck
from models.conditional_entropy_model import SymmetricConditional

################### Compression Network (with factorized entropy model) ###################

def compress_factorized(cubes, model, ckpt_dir):
  """Compress cubes to bitstream.
  Input: cubes with shape [batch size, length, width, height, channel(1)].
  Output: compressed bitstream.
  """

  print('===== Compress =====')
  # load model.
  #model = importlib.import_module(model)
  analysis_transform = model.AnalysisTransform()
  # synthesis_transform = model.SynthesisTransform()
  entropy_bottleneck = EntropyBottleneck()
  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform, 
                                    estimator=entropy_bottleneck)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  x = tf.convert_to_tensor(cubes, "float32")

  def loop_analysis(x):
    x = tf.expand_dims(x, 0)
    y = analysis_transform(x)
    return tf.squeeze(y)

  start = time.time()
  ys = tf.map_fn(loop_analysis, x, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Analysis Transform: {}s".format(round(time.time()-start, 4)))

  start = time.time()
  strings, min_v, max_v = entropy_bottleneck.compress(ys)
  shape = tf.shape(ys)[:]
  print("Entropy Encode: {}s".format(round(time.time()-start, 4)))

  return strings, min_v, max_v, shape

def decompress_factorized(strings, min_v, max_v, shape, model, ckpt_dir):
  """Decompress bitstream to cubes.
  Input: compressed bitstream.
  Output: cubes with shape [batch size, length, width, height, channel(1)]
  """

  print('===== Decompress =====')
  # load model.
  #model = importlib.import_module(model)
  # analysis_transform = model.AnalysisTransform()
  synthesis_transform = model.SynthesisTransform()
  entropy_bottleneck = EntropyBottleneck()
  checkpoint = tf.train.Checkpoint(synthesis_transform=synthesis_transform, 
                                    estimator=entropy_bottleneck)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  start = time.time()
  ys = entropy_bottleneck.decompress(strings, min_v, max_v, shape, shape[-1])
  print("Entropy Decode: {}s".format(round(time.time()-start, 4)))

  def loop_synthesis(y):  
    y = tf.expand_dims(y, 0)
    x = synthesis_transform(y)
    return tf.squeeze(x, [0])

  start = time.time()
  xs = tf.map_fn(loop_synthesis, ys, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Synthesis Transform: {}s".format(round(time.time()-start, 4)))

  return xs

################### Compression Network (with conditional entropy model) ###################

def compress_hyper(cubes, model, ckpt_dir, decompress=False):
  """Compress cubes to bitstream.
  Input: cubes with shape [batch size, length, width, height, channel(1)].
  Output: compressed bitstream.
  """

  print('===== Compress =====')
  # load model.
  #model = importlib.import_module(model)
  analysis_transform = model.AnalysisTransform()
  synthesis_transform = model.SynthesisTransform()
  hyper_encoder = model.HyperEncoder()
  hyper_decoder = model.HyperDecoder()
  entropy_bottleneck = EntropyBottleneck()
  conditional_entropy_model = SymmetricConditional()

  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform, 
                                    synthesis_transform=synthesis_transform, 
                                    hyper_encoder=hyper_encoder, 
                                    hyper_decoder=hyper_decoder, 
                                    estimator=entropy_bottleneck)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  x = tf.convert_to_tensor(cubes, "float32")

  def loop_analysis(x):
    x = tf.expand_dims(x, 0)
    y = analysis_transform(x)
    return tf.squeeze(y)

  start = time.time()
  ys = tf.map_fn(loop_analysis, x, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Analysis Transform: {}s".format(round(time.time()-start, 4)))

  def loop_hyper_encoder(y):
    y = tf.expand_dims(y, 0)
    z = hyper_encoder(y)
    return tf.squeeze(z)

  start = time.time()
  zs = tf.map_fn(loop_hyper_encoder, ys, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Hyper Encoder: {}s".format(round(time.time()-start, 4)))

  z_hats, _ = entropy_bottleneck(zs, False)
  print("Quantize hyperprior.")

  def loop_hyper_deocder(z):
    z = tf.expand_dims(z, 0)
    loc, scale = hyper_decoder(z)
    return tf.squeeze(loc, [0]), tf.squeeze(scale, [0])

  start = time.time()
  locs, scales = tf.map_fn(loop_hyper_deocder, z_hats, dtype=(tf.float32, tf.float32),
                          parallel_iterations=1, back_prop=False)
  lower_bound = 1e-9# TODO
  scales = tf.maximum(scales, lower_bound)
  print("Hyper Decoder: {}s".format(round(time.time()-start, 4)))

  start = time.time()
  z_strings, z_min_v, z_max_v = entropy_bottleneck.compress(zs)
  z_shape = tf.shape(zs)[:]
  print("Entropy Encode (Hyper): {}s".format(round(time.time()-start, 4)))

  start = time.time()
  # y_strings, y_min_v, y_max_v = conditional_entropy_model.compress(ys, locs, scales)
  # y_shape = tf.shape(ys)[:]
  def loop_range_encode(args):
    y, loc, scale = args
    y = tf.expand_dims(y, 0)
    loc = tf.expand_dims(loc, 0)
    scale = tf.expand_dims(scale, 0)
    y_string, y_min_v, y_max_v = conditional_entropy_model.compress(y, loc, scale)
    return y_string, y_min_v, y_max_v

  args = (ys, locs, scales)
  y_strings, y_min_vs, y_max_vs = tf.map_fn(loop_range_encode, args, 
                                            dtype=(tf.string, tf.int32, tf.int32), 
                                            parallel_iterations=1, back_prop=False)
  y_shape = tf.convert_to_tensor(np.insert(tf.shape(ys)[1:].numpy(), 0, 1))

  print("Entropy Encode: {}s".format(round(time.time()-start, 4)))

  if decompress:
    start = time.time()
    def loop_range_decode(args):
      y_string, loc, scale, y_min_v, y_max_v = args
      loc = tf.expand_dims(loc, 0)
      scale = tf.expand_dims(scale, 0)
      y_decoded = conditional_entropy_model.decompress(y_string, loc, scale, y_min_v, y_max_v, y_shape)
      return tf.squeeze(y_decoded, 0)

    args = (y_strings, locs, scales, y_min_vs, y_max_vs)
    y_decodeds = tf.map_fn(loop_range_decode, args, dtype=tf.float32, parallel_iterations=1, back_prop=False)

    print("Entropy Decode: {}s".format(round(time.time()-start, 4)))

    def loop_synthesis(y):
      y = tf.expand_dims(y, 0)
      x = synthesis_transform(y)
      return tf.squeeze(x, [0])

    start = time.time()
    x_decodeds = tf.map_fn(loop_synthesis, y_decodeds, dtype=tf.float32, parallel_iterations=1, back_prop=False)
    print("Synthesis Transform: {}s".format(round(time.time()-start, 4)))
    return y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape, x_decodeds

  return y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape


def decompress_hyper(y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape, model, ckpt_dir):
  """Decompress bitstream to cubes.
  Input: compressed bitstream. latent representations (y) and hyper prior (z).
  Output: cubes with shape [batch size, length, width, height, channel(1)]
  """
  print('===== Decompress =====')
  # load model.
  #model = importlib.import_module(model)
  synthesis_transform = model.SynthesisTransform()
  hyper_encoder = model.HyperEncoder()
  hyper_decoder = model.HyperDecoder()
  entropy_bottleneck = EntropyBottleneck()
  conditional_entropy_model = SymmetricConditional()

  checkpoint = tf.train.Checkpoint(synthesis_transform=synthesis_transform, 
                                    hyper_encoder=hyper_encoder, 
                                    hyper_decoder=hyper_decoder, 
                                    estimator=entropy_bottleneck)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  start = time.time()
  zs = entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape, z_shape[-1])
  print("Entropy Decoder (Hyper): {}s".format(round(time.time()-start, 4)))

  def loop_hyper_deocder(z):
    z = tf.expand_dims(z, 0)
    loc, scale = hyper_decoder(z)
    return tf.squeeze(loc, [0]), tf.squeeze(scale, [0])

  start = time.time()
  locs, scales = tf.map_fn(loop_hyper_deocder, zs, dtype=(tf.float32, tf.float32),
                          parallel_iterations=1, back_prop=False)
  lower_bound = 1e-9# TODO
  scales = tf.maximum(scales, lower_bound)
  print("Hyper Decoder: {}s".format(round(time.time()-start, 4)))

  start = time.time()
  # ys = conditional_entropy_model.decompress(y_strings, locs, scales, y_min_v, y_max_v, y_shape)
  def loop_range_decode(args):
    y_string, loc, scale, y_min_v, y_max_v = args
    loc = tf.expand_dims(loc, 0)
    scale = tf.expand_dims(scale, 0)
    y_decoded = conditional_entropy_model.decompress(y_string, loc, scale, y_min_v, y_max_v, y_shape)
    return tf.squeeze(y_decoded, 0)

  
  args = (y_strings, locs, scales, y_min_vs, y_max_vs)
  ys = tf.map_fn(loop_range_decode, args, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Entropy Decoder: {}s".format(round(time.time()-start, 4)))

  def loop_synthesis(y):
    y = tf.expand_dims(y, 0)
    x = synthesis_transform(y)
    return tf.squeeze(x, [0])

  start = time.time()
  xs = tf.map_fn(loop_synthesis, ys, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Synthesis Transform: {}s".format(round(time.time()-start, 4)))

  return xs

