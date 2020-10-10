# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2019.09.15

from __future__ import print_function
import numpy as np
import os
import random
from pyntcloud import PyntCloud

def read_file(dir, postfix='.obj'):
  filedirs=[]
  def traverse(f):
    fs = os.listdir(f)
    for f1 in fs:
      tmp_path = os.path.join(f, f1)
      if not os.path.isdir(tmp_path):
        filedirs.append(tmp_path)
      else:
        traverse(tmp_path)
  traverse(dir)
  filedirs = [f for f in filedirs if f.endswith(postfix)]
  random.shuffle(filedirs)
  random.shuffle(filedirs)
  return filedirs

def get_rotate_matrix():
    # random rotate
    m=np.eye(3,dtype='float32')
    m[0,0]*=np.random.randint(0,2)*2-1
    m=np.dot(m,np.linalg.qr(np.random.randn(3,3))[0])
    return m

def write_ply_data(filename, data, attribute=False):
  file_rec = open(filename,'a+')
  #print('data.shape:',data.shape)

  file_rec.writelines(['ply\n','format ascii 1.0\n'])
  file_rec.write('element vertex '+str(data.shape[0])+'\n')
  
  if not attribute:
    file_rec.writelines(['property float x\n','property float y\n','property float z\n'])
    file_rec.write('end_header\n')
    for i in range(data.shape[0]):
      file_rec.writelines([str(data[i,0]), ' ', str(data[i,1]), ' ',str(data[i,2]), '\n'])
  else:
    file_rec.writelines(['property float x\n','property float y\n','property float z\n', 
      'property uchar red\n', 'property uchar green\n', 'property uchar blue\n'])
    file_rec.write('end_header\n')
    for i in range(data.shape[0]):
      file_rec.writelines([str(data[i,0]), ' ', str(data[i,1]), ' ',str(data[i,2]), ' ', 
      str(data[i,3]), ' ', str(data[i,4]), ' ',str(data[i,5]), '\n'])   

  file_rec.close() 
  return

def sample_points(root_mesh_dir, root_points_dir, n_points, resolution):
  #1. read folders
  filedirs = read_file(root_mesh_dir, postfix='.obj')[:200]
  print('shapenet:', len(filedirs))

  for index, filedir in enumerate(filedirs):
    # 1. transform format to .ply
    os.system('pcl_obj2ply -format 1 ' + filedir + ' ' + './tp.ply')

    # 2. sample points.
    pc_mesh = PyntCloud.from_file('tp.ply')
    pc = pc_mesh.get_sample("mesh_random", n=n_points, as_PyntCloud=True)

    # 3. random rotate.
    points = pc.points.values
    points = np.dot(points, get_rotate_matrix())

    # 4. voxelization.
    points = points - np.min(points)
    points = points / np.max(points)
    points = points * (resolution)
    points = np.round(points).astype('float32')
    coords = ['x', 'y', 'z']
    pc.points[coords] = points

    if len(set(pc.points.columns) - set(coords)) > 0:
      pc.points = pc.points.groupby(by=coords, sort=False).mean()
    else:
      pc.points = pc.points.drop_duplicates()
    
    # 5. write points.
    pcdir = os.path.join(root_points_dir, str(index)+'.ply')   
    os.system('rm ' + pcdir)
    write_ply_data(pcdir, pc.points.values)


if __name__ == "__main__":
  n_points = int(4e5)
  resolution = 255
  root_mesh_dir = '/home/ubuntu/HardDisk1/ShapeNetCore.v2/02747177/1b7d468a27208ee3dad910e221d16b18/models/'
  root_points_dir = '/home/ubuntu/HardDisk1/shapenet255/'
  if not os.path.exists(root_points_dir):
    os.makedirs(root_points_dir)

  sample_points(root_mesh_dir, root_points_dir, n_points, resolution)

