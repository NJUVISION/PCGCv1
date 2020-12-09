import os
import numpy as np
import sys
sys.path.append('..')
from myutils.gpcc_wrapper import gpcc_encode, gpcc_decode
from dataprocess.inout_points import load_ply_data, write_ply_data

################### bitstream io with out hyper prior ###################

def write_binary_files_factorized(filename, strings, points_numbers, cube_positions, min_v, max_v, shape, rootdir='./'):
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

def read_binary_files_factorized(filename, rootdir='./'):
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

################### bitstream io with hyper prior ###################


def write_binary_files_hyper(filename, y_strings, z_strings, points_numbers, cube_positions, y_min_vs, y_max_vs, y_shape, z_min_v, z_max_v, z_shape, rootdir='./'):
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
    f.write(np.array(y_shape, dtype=np.int16).tobytes())# [batch size, length, width, height, channels]

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


def read_binary_files_hyper(filename, rootdir='./'):
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
    y_shape = np.frombuffer(f.read(2*5), dtype=np.int16)

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
  
  # y_shape = np.array([1, 16, 16, 16, 16])
  return y_strings, z_strings, points_numbers, cube_positions, y_min_vs, y_max_vs, y_shape, z_min_v, z_max_v, z_shape

