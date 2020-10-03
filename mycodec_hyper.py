# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Hao Zhu, Zhan Ma, Tong Chen, Haojie Liu, Qiu Shen; Nanjing University, Vision Lab.
# Chaofei Wang; Shanghai Jiao Tong University, Cooperative Medianet Innovation Center.

# Last update: 
# 2019.11.13 compress/decompress required less GPU memory.
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
from dataprocess.inout_points import load_ply_data, write_ply_data, load_points, save_points, points2voxels, voxels2points
from dataprocess.post_process import select_voxels
from myutils.gpcc_wrapper import gpcc_encode, gpcc_decode

###################################### Preprocess & Postprocess ######################################

def preprocess(input_file, scale, cube_size, min_num):
  """Scaling, Partition & Voxelization.
  Input: .ply file and arguments for pre-process.  
  Output: partitioned cubes, cube positions, and number of points in each cube. 
  """

  print('===== Preprocess =====')
  # scaling (optional)
  start = time.time()
  if scale == 1:
    scaling_file = input_file 
  else:
    pc = load_ply_data(input_file)
    pc_down = np.round(pc.astype('float32') * scale)
    pc_down = np.unique(pc_down, axis=0)# remove duplicated points
    scaling_file = './downscaling.ply'
    write_ply_data(scaling_file, pc_down)
  print("Scaling: {}s".format(round(time.time()-start, 4)))

  # partition.
  start = time.time()
  partitioned_points, cube_positions = load_points(scaling_file, cube_size, min_num)
  print("Partition: {}s".format(round(time.time()-start, 4)))

  # voxelization.
  start = time.time()
  cubes = points2voxels(partitioned_points, cube_size)
  points_numbers = np.sum(cubes, axis=(1,2,3,4)).astype(np.uint16)
  print("Voxelization: {}s".format(round(time.time()-start, 4)))

  print('cubes shape: {}'.format(cubes.shape))
  print('points numbers (sum/mean/max/min): {} {} {} {}'.format( 
  points_numbers.sum(), round(points_numbers.mean()), points_numbers.max(), points_numbers.min()))

  return cubes, cube_positions, points_numbers

def postprocess(output_file, cubes, points_numbers, cube_positions, scale, cube_size, rho):
  """Classify voxels to occupied or free, then extract points and write to file.
  Input:  deocded cubes, cube positions, points numbers, cube size and rho=ouput numbers/input numbers.
  """

  print('===== Post process =====')
  # Classify.
  start = time.time()
  output = select_voxels(cubes, points_numbers, rho)
  
  # Extract points.
  #points = voxels2points(output.numpy())
  points = voxels2points(output)
  print("Classify and extract points: {}s".format(round(time.time()-start, 4)))

  # scaling (optional)
  start = time.time()
  if scale == 1:
    save_points(points, cube_positions, output_file, cube_size)
  else:
    scaling_output_file = './downsampling_rec.ply'
    save_points(points, cube_positions, scaling_output_file, cube_size)
    pc = load_ply_data(scaling_output_file)
    pc_up = pc.astype('float32') * float(1/scale)
    write_ply_data(output_file, pc_up)
  print("Write point cloud to {}: {}s".format(output_file, round(time.time()-start, 4)))

  return

###################################### Compress & Decompress ######################################

def compress(cubes, model, ckpt_dir, decompress=False):
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
  y_strings, y_min_v, y_max_v = conditional_entropy_model.compress(ys, locs, scales)
  y_shape = tf.shape(ys)[:]
  print("Entropy Encode: {}s".format(round(time.time()-start, 4)))

  if decompress:
    # y_hats, _ = conditional_entropy_model(ys, locs, scales, False)
    start = time.time()
    y_decodeds = conditional_entropy_model.decompress(y_strings, locs, scales, y_min_v, y_max_v, y_shape)
    print("Entropy Decode: {}s".format(round(time.time()-start, 4)))

    def loop_synthesis(y):
      y = tf.expand_dims(y, 0)
      x = synthesis_transform(y)
      return tf.squeeze(x, [0])

    start = time.time()
    x_decodeds = tf.map_fn(loop_synthesis, y_decodeds, dtype=tf.float32, parallel_iterations=1, back_prop=False)
    print("Synthesis Transform: {}s".format(round(time.time()-start, 4)))

    return y_strings, y_min_v, y_max_v, y_shape, z_strings, z_min_v, z_max_v, z_shape, x_decodeds

  return y_strings, y_min_v, y_max_v, y_shape, z_strings, z_min_v, z_max_v, z_shape

def decompress(y_strings, y_min_v, y_max_v, y_shape, z_strings, z_min_v, z_max_v, z_shape, model, ckpt_dir):
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
  ys = conditional_entropy_model.decompress(y_strings, locs, scales, y_min_v, y_max_v, y_shape)
  print("Entropy Decoder: {}s".format(round(time.time()-start, 4)))

  def loop_synthesis(y):
    y = tf.expand_dims(y, 0)
    x = synthesis_transform(y)
    return tf.squeeze(x, [0])

  start = time.time()
  xs = tf.map_fn(loop_synthesis, ys, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Synthesis Transform: {}s".format(round(time.time()-start, 4)))

  return xs

# ###################################### losslessly compress cube positions using MPEG G-PCC. ######################################

# def gpcc_encode(filedir, bin_dir, show=False):
#   """Compress point cloud losslessly using MPEG G-PCCv6. 
#   You can download and install TMC13 from 
#   http://mpegx.int-evry.fr/software/MPEG/PCC/TM/mpeg-pcc-tmc13
#   """

#   subp=subprocess.Popen('../../mpeg/mpeg-pcc-tmc13/build/tmc3/tmc3'+ 
#                         ' --mode=0' + 
#                         ' --positionQuantizationScale=1' + 
#                         ' --trisoup_node_size_log2=0' + 
#                         ' --ctxOccupancyReductionFactor=3' + 
#                         ' --neighbourAvailBoundaryLog2=8' + 
#                         ' --intra_pred_max_node_size_log2=6' + 
#                         ' --inferredDirectCodingMode=0' + 
#                         ' --uncompressedDataPath='+filedir + 
#                         ' --compressedStreamPath='+bin_dir, 
#                         shell=True, stdout=subprocess.PIPE)
#   c=subp.stdout.readline()
#   while c:
#     if show:
#       print(c)
#     c=subp.stdout.readline()
  
#   return 

# def gpcc_decode(bin_dir, rec_dir, show=False):
#   subp=subprocess.Popen('../../mpeg/mpeg-pcc-tmc13/build/tmc3/tmc3'+ 
#                         ' --mode=1'+ 
#                         ' --compressedStreamPath='+bin_dir+ 
#                         ' --reconstructedDataPath='+rec_dir, 
#                         shell=True, stdout=subprocess.PIPE)
#   c=subp.stdout.readline()
#   while c:
#     if show:
#       print(c)      
#     c=subp.stdout.readline()
  
#   return

###################################### write & read binary files. ######################################

def write_binary_files(filename, y_strings, z_strings, points_numbers, cube_positions, y_min_v, y_max_v, y_shape, z_min_v, z_max_v, z_shape, rootdir='./'):
  """Write compressed binary files:
    1) Compressed latent features.
    2) Compressed hyperprior.
    3) Number of input points.
    4) Positions of each cube.
  """ 

  if not os.path.exists(rootdir):
    os.makedirs(rootdir)
  print('===== Write binary files =====')
  file_strings = os.path.join(rootdir, filename+'.strings')
  file_strings_hyper = os.path.join(rootdir, filename+'.strings_hyper')
  file_pointnums = os.path.join(rootdir, filename+'.pointnums')
  file_cubepos = os.path.join(rootdir, filename+'.cubepos')
  ply_cubepos = os.path.join(rootdir, filename+'_cubepos.ply')
  
  with open(file_strings, 'wb') as f:
    f.write(np.array(y_shape, dtype=np.int16).tobytes())# [batch size, length, width, height, channels]
    f.write(np.array((y_min_v, y_max_v), dtype=np.int8).tobytes())
    f.write(y_strings)

  with open(file_strings_hyper, 'wb') as f:
    f.write(np.array(z_shape, dtype=np.int16).tobytes())# [batch size, length, width, height, channels]
    f.write(np.array((z_min_v, z_max_v), dtype=np.int8).tobytes())
    f.write(z_strings)

  # TODO: Compress numbers of points.
  with open(file_pointnums, 'wb') as f:
    f.write(np.array(points_numbers, dtype=np.uint16).tobytes())
  
  write_ply_data(ply_cubepos, cube_positions.astype('uint8'))
  gpcc_encode(ply_cubepos, file_cubepos)
  
  bytes_strings = os.path.getsize(file_strings)
  bytes_strings_hyper = os.path.getsize(file_strings_hyper)
  bytes_pointnums = os.path.getsize(file_pointnums)
  bytes_cubepos = os.path.getsize(file_cubepos)
  print('Total file size (Bytes): {}'.format(bytes_strings+bytes_strings_hyper+bytes_pointnums+bytes_cubepos))
  print('Strings (Bytes): {}'.format(bytes_strings))
  print('Strings hyper (Bytes): {}'.format(bytes_strings_hyper))
  print('Numbers of points (Bytes): {}'.format(bytes_pointnums))
  print('Positions of cubes (Bytes): {}'.format(bytes_cubepos))

  return bytes_strings, bytes_strings_hyper, bytes_pointnums, bytes_cubepos

def read_binary_files(filename, rootdir='./'):
  """Read from compressed binary files:
    1) Compressed latent features.
    2) Compressed hyperprior.
    3) Number of input points.
    4) Positions of each cube.
  """ 

  print('===== Read binary files =====')
  file_strings = os.path.join(rootdir, filename+'.strings')
  file_strings_hyper = os.path.join(rootdir, filename+'.strings_hyper')
  file_pointnums = os.path.join(rootdir, filename+'.pointnums')
  file_cubepos = os.path.join(rootdir, filename+'.cubepos')
  ply_cubepos = os.path.join(rootdir, filename+'_cubepos.ply')
  
  with open(file_strings, 'rb') as f:
    y_shape = np.frombuffer(f.read(2*5), dtype=np.int16)
    y_min_v, y_max_v = np.frombuffer(f.read(1*2), dtype=np.int8)
    y_strings = f.read()

  with open(file_strings_hyper, 'rb') as f:
    z_shape = np.frombuffer(f.read(2*5), dtype=np.int16)
    z_min_v, z_max_v = np.frombuffer(f.read(1*2), dtype=np.int8)
    z_strings = f.read()

  with open(file_pointnums, 'rb') as f:
    points_numbers = np.frombuffer(f.read(), dtype=np.uint16)
  
  gpcc_decode(file_cubepos, ply_cubepos)
  cube_positions = load_ply_data(ply_cubepos)
  
  return y_strings, z_strings, points_numbers, cube_positions, y_min_v, y_max_v, y_shape, z_min_v, z_max_v, z_shape

###################################### Compress & Decompress required less GPU memory. ######################################
# Each cube is arithmetric coded (range code) separately, so it required less GPU memory.
# You may need this when process large point cloud or at high bit rate.

def compress_less_mem(cubes, model, ckpt_dir, decompress=False):
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
    # y_hats, _ = conditional_entropy_model(ys, locs, scales, False)
    start = time.time()
    # y_decodeds = conditional_entropy_model.decompress(y_strings, locs, scales, y_min_v, y_max_v, y_shape)
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

def compress_more_less_mem(cubes, model, ckpt_dir, decompress=False):
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

    start = time.time()
    y_strings = [] 
    y_min_vs = [] 
    y_max_vs = []
    x_decodeds = []
    zs = []
    for idx, cube in enumerate(cubes):
        if idx % 1000 == 0:
            print(idx)
        x = tf.convert_to_tensor(cube, "float32")
        x = tf.expand_dims(x, 0)
        y = analysis_transform(x)
        z = hyper_encoder(y)
        zs.append(z)
        z_hat, _ = entropy_bottleneck(z, False)
        loc, scale = hyper_decoder(z_hat)
        lower_bound = 1e-9# TODO
        scale = tf.maximum(scale, lower_bound)
        y_string, y_min_v, y_max_v = conditional_entropy_model.compress(y, loc, scale)
        y_strings.append(y_string)
        y_min_vs.append(y_min_v)
        y_max_vs.append(y_max_v)

        y_shape = tf.shape(y)
        y_dec = conditional_entropy_model.decompress(y_string, loc, scale, y_min_v, y_max_v, y_shape)
        x_dec = synthesis_transform(y_dec)
        x_dec = x_dec.numpy()
        x_decodeds.append(x_dec)
 
    y_strings = tf.convert_to_tensor(y_strings, dtype='string')    
    y_min_vs = tf.convert_to_tensor(y_min_vs)
    y_max_vs = tf.convert_to_tensor(y_max_vs)
    zs = tf.concat(zs, axis=0)
    x_decodeds = np.concatenate(x_decodeds, 0)
    x_decodeds = tf.convert_to_tensor(x_decodeds, 'float32')

    z_strings, z_min_v, z_max_v = entropy_bottleneck.compress(zs)
    z_shape = tf.shape(zs)[:]


    return y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape, x_decodeds



def decompress_less_mem(y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape, model, ckpt_dir):
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

###################################### write & read binary file for compression required less menory. ######################################

def write_binary_files_less_mem(filename, y_strings, z_strings, points_numbers, cube_positions, y_min_vs, y_max_vs, y_shape, z_min_v, z_max_v, z_shape, rootdir='./'):
  """Write compressed binary files:
    1) Compressed latent features.
    2) Compressed hyperprior.
    3) Number of input points.
    4) Positions of each cube.
  """ 

  if not os.path.exists(rootdir):
    os.makedirs(rootdir)
  print('===== Write binary files =====')
  file_strings = os.path.join(rootdir, filename+'.strings')
  file_strings_head = os.path.join(rootdir, filename+'.strings_head')
  file_strings_hyper = os.path.join(rootdir, filename+'.strings_hyper')
  file_pointnums = os.path.join(rootdir, filename+'.pointnums')
  file_cubepos = os.path.join(rootdir, filename+'.cubepos')
  ply_cubepos = os.path.join(rootdir, filename+'_cubepos.ply')

  with open(file_strings_head, 'wb') as f:
    f.write(np.array(len(y_strings), dtype=np.int16).tobytes())
    y_max_min_vs = y_max_vs*16 - y_min_vs
    f.write(np.array(y_max_min_vs, dtype=np.uint8).tobytes())
    y_strings_lens = np.array([len(y_string) for _, y_string in enumerate(y_strings)])
    for i , l in enumerate(y_strings_lens):
      if l <= 255:
        f.write(np.array(l, dtype=np.uint8).tobytes())
      else:
        f.write(np.array(0, dtype=np.uint8).tobytes())
        f.write(np.array(l, dtype=np.int16).tobytes())

  with open(file_strings, 'wb') as f:
    for i, y_string in enumerate(y_strings):
      f.write(y_string)
  
  with open(file_strings_hyper, 'wb') as f:
    f.write(np.array(z_shape, dtype=np.int16).tobytes())# [batch size, length, width, height, channels]
    f.write(np.array((z_min_v, z_max_v), dtype=np.int8).tobytes())
    f.write(z_strings)

  # TODO: Compress numbers of points.
  with open(file_pointnums, 'wb') as f:
    f.write(np.array(points_numbers, dtype=np.uint16).tobytes())
  
  write_ply_data(ply_cubepos, cube_positions.astype('uint8'))
  gpcc_encode(ply_cubepos, file_cubepos)
  
  # bytes_strings = sum([os.path.getsize(f) for f in glob.glob(file_strings_folder+'/*.strings')])
  bytes_strings = os.path.getsize(file_strings)
  bytes_strings_head = os.path.getsize(file_strings_head)
  bytes_strings_hyper = os.path.getsize(file_strings_hyper)
  bytes_pointnums = os.path.getsize(file_pointnums)
  bytes_cubepos = os.path.getsize(file_cubepos)

  print('Total file size (Bytes): {}'.format(bytes_strings + 
                                            bytes_strings_head + 
                                            bytes_strings_hyper + 
                                            bytes_pointnums + 
                                            bytes_cubepos))

  print('Strings (Bytes): {}'.format(bytes_strings))
  print('Strings head (Bytes): {}'.format(bytes_strings_head))
  print('Strings hyper (Bytes): {}'.format(bytes_strings_hyper))
  print('Numbers of points (Bytes): {}'.format(bytes_pointnums))
  print('Positions of cubes (Bytes): {}'.format(bytes_cubepos))

  return bytes_strings, bytes_strings_head, bytes_strings_hyper, bytes_pointnums, bytes_cubepos


def read_binary_files_less_mem(filename, rootdir='./'):
  """Read from compressed binary files:
    1) Compressed latent features.
    2) Compressed hyperprior.
    3) Number of input points.
    4) Positions of each cube.
  """ 

  print('===== Read binary files =====')
  file_strings = os.path.join(rootdir, filename+'.strings')
  file_strings_head = os.path.join(rootdir, filename+'.strings_head')
  file_strings_hyper = os.path.join(rootdir, filename+'.strings_hyper')
  file_pointnums = os.path.join(rootdir, filename+'.pointnums')
  file_cubepos = os.path.join(rootdir, filename+'.cubepos')
  ply_cubepos = os.path.join(rootdir, filename+'_cubepos.ply')

  with open(file_strings_head, 'rb') as f:
    y_strings_num = int(np.frombuffer(f.read(1*2), dtype=np.int16))
    y_max_min_vs = np.frombuffer(f.read(y_strings_num*1), dtype=np.uint8).astype('int32')
    y_max_vs = y_max_min_vs // 16
    y_min_vs = -(y_max_min_vs % 16)
    # y_min_vs = np.array(y_min_vs).astype('int32')
    # y_max_vs = np.array(y_max_vs).astype('int32')

    y_strings_lens = []
    for i in range(y_strings_num):
      l = np.frombuffer(f.read(1*1), dtype=np.uint8)
      if l == 0:
        l = int(np.frombuffer(f.read(1*2), dtype=np.int16))
      y_strings_lens.append(l)
    y_strings_lens = np.array(y_strings_lens, dtype=np.int32)
    f.close()

  with open(file_strings, 'rb') as f:
    y_strings = []
    for i, l in enumerate(y_strings_lens):
      y_strings.append(f.read(int(l)))
    y_strings = np.array(y_strings)
    f.close()

  with open(file_strings_hyper, 'rb') as f:
    z_shape = np.frombuffer(f.read(2*5), dtype=np.int16)
    z_min_v, z_max_v = np.frombuffer(f.read(1*2), dtype=np.int8)
    z_strings = f.read()

  with open(file_pointnums, 'rb') as f:
    points_numbers = np.frombuffer(f.read(), dtype=np.uint16)
  
  gpcc_decode(file_cubepos, ply_cubepos)
  cube_positions = load_ply_data(ply_cubepos)
  
  y_shape = np.array([1, 16, 16, 16, 16])
  return y_strings, z_strings, points_numbers, cube_positions, y_min_vs, y_max_vs, y_shape, z_min_v, z_max_v, z_shape


def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "command", choices=["compress", "decompress"],
      help="What to do: 'compress' reads a point cloud (.ply format) "
          "and writes compressed binary files. 'decompress' "
          "reads binary files and reconstructs the point cloud (.ply format). "
          "input and output filenames need to be provided for the latter. ")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")

    # parser.add_argument(
    #   "--model", default="model_voxception",
    #   help="model.")
  parser.add_argument(
    "--ckpt_dir", type=str, default='', dest="ckpt_dir",
    help='checkpoint direction trained with different RD tradeoff')

  parser.add_argument(
      "--scale", type=float, default=1.0, dest="scale",
      help="point cloud scaling factor.")
  parser.add_argument(
      "--cube_size", type=int, default=64, dest="cube_size",
      help="size of partitioned cubes.")
  parser.add_argument(
      "--min_num", type=int, default=100, dest="min_num",
      help="minimum number of points in a cube.")
  parser.add_argument(
      "--rho", type=float, default=1.0, dest="rho",
      help="ratio of the numbers of output points to the number of input points.")

  parser.add_argument(
      "--less_mem", type=int, default=0, dest="less_mem",
      help="whether compress required less memory? (0/1)")

  parser.add_argument(
      "--gpu", type=int, default=1, dest="gpu",
      help="use gpu (1) or not (0).")
  args = parser.parse_args()
  print(args)

  return args

if __name__ == "__main__":
  """
  Examples:
  python mycodec_hyper.py compress 'testdata/8iVFB/longdress_vox10_1300.ply' --ckpt_dir='checkpoints/hyper/a0.75b3/' --gpu=1

  python mycodec_hyper.py decompress 'compressed/longdress_vox10_1300' --ckpt_dir='checkpoints/hyper/a0.75b3/' --gpu=1
  """

  args = parse_args()
  if args.gpu==1:
    os.environ['CUDA_VISIBLE_DEVICES']="0"
  else:
    os.environ['CUDA_VISIBLE_DEVICES']=""
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 1.0
  config.gpu_options.allow_growth = True
  config.log_device_placement=True
  sess = tf.Session(config=config)

  if args.command == "compress":
    if not args.output:
      args.output = os.path.split(args.input)[-1][:-4]

    cubes, cube_positions, points_numbers = preprocess(args.input, args.scale, args.cube_size, args.min_num)
    
    if args.less_mem == 0:
      y_strings, y_min_v, y_max_v, y_shape, z_strings, z_min_v, z_max_v, z_shape = compress(cubes, model, args.ckpt_dir)

      bytes_strings, bytes_strings_hyper, bytes_pointnums, bytes_cubepos = write_binary_files(
          args.output, y_strings.numpy(), z_strings.numpy(), points_numbers, cube_positions,
          y_min_v.numpy(), y_max_v.numpy(), y_shape.numpy(), 
          z_min_v.numpy(), z_max_v.numpy(), z_shape.numpy(), rootdir='./compressed')
      
    elif args.less_mem == 1:
      y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape = compress_less_mem(cubes, model, args.ckpt_dir)

      bytes_strings, bytes_strings_head, bytes_strings_hyper, bytes_pointnums, bytes_cubepos = write_binary_files_less_mem(
          args.output, y_strings.numpy(), z_strings.numpy(), points_numbers, cube_positions,
          y_min_vs.numpy(), y_max_vs.numpy(), y_shape.numpy(), 
          z_min_v.numpy(), z_max_v.numpy(), z_shape.numpy(), rootdir='./compressed')


  elif args.command == "decompress":
    rootdir, filename = os.path.split(args.input)
    if not args.output:
      args.output = filename + "_rec.ply"

    if args.less_mem == 0:
      y_strings_d, z_strings_d, points_numbers_d, cube_positions_d, \
      y_min_v_d, y_max_v_d, y_shape_d, z_min_v_d, z_max_v_d, z_shape_d = read_binary_files(filename, rootdir)

      cubes_d = decompress(y_strings_d, y_min_v_d, y_max_v_d, y_shape_d, z_strings_d, z_min_v_d, z_max_v_d, z_shape_d, model, args.ckpt_dir)
    
    elif args.less_mem == 1:
      y_strings_d, z_strings_d, points_numbers_d, cube_positions_d, \
      y_min_vs_d, y_max_vs_d, y_shape_d, z_min_v_d, z_max_v_d, z_shape_d = read_binary_files_less_mem(filename, rootdir)

      cubes_d = decompress_less_mem(y_strings_d, y_min_vs_d, y_max_vs_d, y_shape_d, z_strings_d, z_min_v_d, z_max_v_d, z_shape_d, model, args.ckpt_dir)
    
    postprocess(args.output, cubes_d, points_numbers_d, cube_positions_d, args.scale, args.cube_size, args.rho)
    