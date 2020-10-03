# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Hao Zhu, Zhan Ma, Tong Chen, Haojie Liu, Qiu Shen; Nanjing University, Vision Lab.
# Chaofei Wang; Shanghai Jiao Tong University, Cooperative Medianet Innovation Center.

# Last update: 
# 2019.10.07

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
from dataprocess.inout_points import load_ply_data, write_ply_data, load_points, save_points, points2voxels, voxels2points
from dataprocess.post_process import select_voxels
from myutils.gpcc_wrapper import gpcc_encode, gpcc_decode

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
    # pc_up = pc.astype('float32') * scale
    pc_up = pc.astype('float32') * 1/float(scale)#
    write_ply_data(output_file, pc_up)
  print("Write point cloud to {}: {}s".format(output_file, round(time.time()-start, 4)))

  return
  

def compress(cubes, model, ckpt_dir):
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

def decompress(strings, min_v, max_v, shape, model, ckpt_dir):
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


def write_binary_files(filename, strings, points_numbers, cube_positions, min_v, max_v, shape, rootdir='./'):
  """Write compressed binary files:
    1) Compressed latent features.
    2) Number of input points.
    3) Positions of each cube.
  """ 
  if not os.path.exists(rootdir):
    os.makedirs(rootdir)
  print('===== Write binary files =====')
  file_strings = os.path.join(rootdir, filename+'.strings')
  file_pointnums = os.path.join(rootdir, filename+'.pointnums')
  file_cubepos = os.path.join(rootdir, filename+'.cubepos')
  ply_cubepos = os.path.join(rootdir, filename+'_cubepos.ply')
  
  with open(file_strings, 'wb') as f:
    f.write(np.array(shape, dtype=np.int16).tobytes())# [batch size, length, width, height, channels]
    f.write(np.array((min_v, max_v), dtype=np.int8).tobytes())
    f.write(strings)

  # TODO: Compress numbers of points.
  with open(file_pointnums, 'wb') as f:
    f.write(np.array(points_numbers, dtype=np.uint16).tobytes())
  
  write_ply_data(ply_cubepos, cube_positions.astype('uint8'))
  gpcc_encode(ply_cubepos, file_cubepos)
  
  bytes_strings = os.path.getsize(file_strings)
  bytes_pointnums = os.path.getsize(file_pointnums)
  bytes_cubepos = os.path.getsize(file_cubepos)
  print('Total file size (Bytes): {}'.format(bytes_strings+bytes_pointnums+bytes_cubepos))
  print('Strings (Bytes): {}'.format(bytes_strings))
  print('Numbers of points (Bytes): {}'.format(bytes_pointnums))
  print('Positions of cubes (Bytes): {}'.format(bytes_cubepos))

  return bytes_strings, bytes_pointnums, bytes_cubepos

def read_binary_files(filename, rootdir='./'):
  """Read from compressed binary files:
    1) Compressed latent features.
    2) Number of input points.
    3) Positions of each cube.
  """ 

  print('===== Read binary files =====')
  file_strings = os.path.join(rootdir, filename+'.strings')
  file_pointnums = os.path.join(rootdir, filename+'.pointnums')
  file_cubepos = os.path.join(rootdir, filename+'.cubepos')
  ply_cubepos = os.path.join(rootdir, filename+'_cubepos.ply')
  
  with open(file_strings, 'rb') as f:
    shape = np.frombuffer(f.read(2*5), dtype=np.int16)
    min_v, max_v = np.frombuffer(f.read(1*2), dtype=np.int8)
    strings = f.read()

  with open(file_pointnums, 'rb') as f:
    points_numbers = np.frombuffer(f.read(), dtype=np.uint16)
  
  gpcc_decode(file_cubepos, ply_cubepos)
  cube_positions = load_ply_data(ply_cubepos)
  
  return strings, points_numbers, cube_positions, min_v, max_v, shape


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
      "--min_num", type=int, default=20, dest="min_num",
      help="minimum number of points in a cube.")
  parser.add_argument(
      "--rho", type=float, default=1.0, dest="rho",
      help="ratio of the numbers of output points to the number of input points.")

  parser.add_argument(
      "--gpu", type=int, default=1, dest="gpu",
      help="use gpu (1) or not (0).") 
  args = parser.parse_args()
  print(args)

  return args

if __name__ == "__main__":
  """
  Examples:
  python mycodec_factorized.py compress 'testdata/8iVFB/loot_vox10_1200.ply' --ckpt_dir='checkpoints/factorized/a2b3/' --gpu=1

  python mycodec_factorized.py decompress 'compressed/loot_vox10_1200' --ckpt_dir='checkpoints/factorized/a2b3/' --gpu=1
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
    cubes, cube_positions, points_numbers = preprocess(args.input, args.scale, args.cube_size, args.min_num)
    strings, min_v, max_v, shape = compress(cubes, model, args.ckpt_dir)
    if not args.output:
      args.output = os.path.split(args.input)[-1][:-4]
    bytes_strings, bytes_pointnums, bytes_cubepos = write_binary_files(
      args.output, strings.numpy(), points_numbers, cube_positions, min_v.numpy(), max_v.numpy(), shape.numpy(), rootdir='./compressed')

  elif args.command == "decompress":
    rootdir, filename = os.path.split(args.input)
    if not args.output:
      args.output = filename + "_rec.ply"
    strings_d, points_numbers_d, cube_positions_d, min_v_d, max_v_d, shape_d = read_binary_files(filename, rootdir)
    cubes_d = decompress(strings_d, min_v_d, max_v_d, shape_d, model, args.ckpt_dir)
    postprocess(args.output, cubes_d, points_numbers_d, cube_positions_d, args.scale, args.cube_size, args.rho)
