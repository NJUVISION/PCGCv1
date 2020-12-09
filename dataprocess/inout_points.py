# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2019.09.17

import numpy as np
import os

################### plyfile <--> points ####################
def load_ply_data(filename):
  '''
  load data from ply file.
  '''
  f = open(filename)
  #1.read all points
  points = []
  for line in f:
    #only x,y,z
    wordslist = line.split(' ')
    try:
      x, y, z = float(wordslist[0]),float(wordslist[1]),float(wordslist[2])
    except ValueError:
      continue
    points.append([x,y,z])
  points = np.array(points)
  points = points.astype(np.int32)#np.uint8
  # print(filename,'\n','length:',points.shape)
  f.close()

  return points

def write_ply_data(filename, points):
  '''
  write data to ply file.
  '''
  if os.path.exists(filename):
    os.system('rm '+filename)
  f = open(filename,'a+')
  #print('data.shape:',data.shape)
  f.writelines(['ply\n','format ascii 1.0\n'])
  f.write('element vertex '+str(points.shape[0])+'\n')
  f.writelines(['property float x\n','property float y\n','property float z\n'])
  f.write('end_header\n')
  for _, point in enumerate(points):
    f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), '\n'])
  f.close() 

  return

#################### plyfile <--> partition points ####################

def load_points(filename, cube_size=64, min_num=20):
  """Load point cloud & split to cubes.
  
  Args: point cloud file; voxel size; minimun number of points in a cube.

  Return: cube positions & points in each cube.
  """

  # load point clouds
  point_cloud = load_ply_data(filename)
  # partition point cloud to cubes.
  cubes = {}# {block start position, points in block}
  for _, point in enumerate(point_cloud):
    cube_index = tuple((point//cube_size).astype("int"))
    local_point = point % cube_size
    if not cube_index in cubes.keys():
      cubes[cube_index] = local_point
    else:
      cubes[cube_index] = np.vstack((cubes[cube_index] ,local_point))
  # filter by minimum number.
  k_del = []
  for _, k in enumerate(cubes.keys()):
    if cubes[k].shape[0] < min_num:
      k_del.append(k)
  for _, k in enumerate(k_del):
    del cubes[k]
  # get points and cube positions.
  cube_positions = np.array(list(cubes.keys()))
  set_points = []
  # orderd
  step = cube_positions.max() + 1
  cube_positions_n = cube_positions[:,0:1] + cube_positions[:,1:2]*step + cube_positions[:,2:3]*step*step
  cube_positions_n = np.sort(cube_positions_n, axis=0)
  x = cube_positions_n % step
  y = (cube_positions_n // step) % step
  z = cube_positions_n // step // step
  cube_positions_orderd = np.concatenate((x,y,z), -1)
  for _, k in enumerate(cube_positions_orderd):
    set_points.append(cubes[tuple(k)].astype("int16"))

  return set_points, cube_positions

def save_points(set_points, cube_positions, filename, cube_size=64):
  """Combine & save points."""

  # order cube positions.
  step = cube_positions.max() + 1
  cube_positions_n = cube_positions[:,0:1] + cube_positions[:,1:2]*step + cube_positions[:,2:3]*step*step
  cube_positions_n = np.sort(cube_positions_n, axis=0)
  x = cube_positions_n % step
  y = (cube_positions_n // step) % step
  z = cube_positions_n // step // step
  cube_positions_orderd = np.concatenate((x,y,z), -1)
  # combine points.
  point_cloud = []
  for k, v in zip(cube_positions_orderd, set_points):
    points = v + np.array(k) * cube_size
    point_cloud.append(points)
  point_cloud = np.concatenate(point_cloud).astype("int")
  
  write_ply_data(filename, point_cloud)

  return

#################### points <--> volumetric models ####################

def points2voxels(set_points, cube_size):
  """Transform points to voxels (binary occupancy map).
  Args: points list; cube size;

  Return: A tensor with shape [batch_size, cube_size, cube_size, cube_size, 1]
  """

  voxels = []
  for _, points in enumerate(set_points):
    points = points.astype("int")
    vol = np.zeros((cube_size,cube_size,cube_size))
    vol[points[:,0],points[:,1],points[:,2]] = 1.0
    vol = np.expand_dims(vol,-1) 
    voxels.append(vol)
  voxels = np.array(voxels)

  return voxels

def voxels2points(voxels):
  """extract points from each voxel."""

  voxels = np.squeeze(np.uint8(voxels)) # 0 or 1
  set_points = []
  for _, vol in enumerate(voxels):
    points = np.array(np.where(vol>0)).transpose((1,0))
    set_points.append(points)
  
  return set_points

#################### select top-k voxels of volumetric model ####################

def select_voxels(vols, points_nums, offset_ratio=1.0, fixed_thres=None):
  '''Select the top k voxels and generate the mask.
  input:  vols: [batch_size, vsize, vsize, vsize, 1] float32
          points numbers: [batch_size]
  output: the mask (0 or 1) representing the selected voxels: [batch_size, vsize, vsize, vsize]  
  '''
  # vols = vols.numpy()
  # points_nums = points_nums
  # offset_ratio = offset_ratio
  masks = []
  for idx, vol in enumerate(vols):
    if fixed_thres==None:
      num = int(offset_ratio * np.array(points_nums[idx]))
      thres = get_adaptive_thres(vol, num)
    else:
      thres = fixed_thres
    # print(thres)
    # mask = np.greater(vol, thres).astype('float32')
    mask = np.greater_equal(vol, thres).astype('float32')
    masks.append(mask)

  return np.stack(masks)

def get_adaptive_thres(vol, num, init_thres=-2.0):
  values = vol[vol>init_thres]
  # number of values should be larger than expected number.
  if values.shape[0] < num:
    values = np.reshape(vol, [-1])
  # only sort the selected values.
  values.sort()
  thres = values[-num]

  return thres


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "--input", type=str, default='../../testdata/8iVFB/redandblack_vox10_1550_n.ply', dest="input")
  parser.add_argument(
    "--output", type=str, default='rec.ply', dest="output")
  parser.add_argument(
    "--cube_size", type=int, default=64, dest="cube_size",
    help="size of partitioned cubes.")
  parser.add_argument(
    "--min_num", type=int, default=20, dest="min_num",
    help="minimum number of points in a cube.")
  args = parser.parse_args()
  print(args)

  ################### test top-k voxels selection #########################
  data = np.random.rand(4, 64, 64, 64, 1) * (100) -50
  points_nums = np.array([1000, 200, 10000, 50])
  offset_ratio = 1.0 
  init_thres = -1.0
  mask = select_voxels(data, points_nums, offset_ratio, init_thres)   
  print(mask.shape)

  ################### inout #########################
  set_points, cube_positions = load_points(args.input, 
                                          cube_size=args.cube_size, 
                                          min_num=args.min_num)
  voxels = points2voxels(set_points, cube_size=args.cube_size)
  print('voxels:',voxels.shape)
  points_rec = voxels2points(voxels)
  save_points(points_rec, cube_positions, args.output, cube_size=args.cube_size)
  os.system("../myutils/pc_error_d" \
  + ' -a ' + args.input + ' -b ' + args.output + " -r 1023")
  os.system("rm "+args.output)
