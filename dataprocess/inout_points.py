# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Hao Zhu, Zhan Ma, Tong Chen, Haojie Liu, Qiu Shen; Nanjing University, Vision Lab.
# Chaofei Wang; Shanghai Jiao Tong University, Cooperative Medianet Innovation Center.
# Last update: 2019.09.17

import numpy as np
import os


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


if __name__=='__main__':
  name = '../testdata/8iVFB/loot_vox10_1200.ply'
  name_rec = 'rec.ply'
  set_points, cube_positions = load_points(name, cube_size=64, min_num=20)
  voxels = points2voxels(set_points, cube_size=64)
  print('voxels:',voxels.shape)
  points_rec = voxels2points(voxels)
  save_points(points_rec, cube_positions, name_rec, cube_size=64)
  os.system("../myutils/pc_error_d" \
  + ' -a ' + name + ' -b ' + name_rec + " -r 1023")
