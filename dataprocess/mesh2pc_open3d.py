#!/usr/bin/env python
# coding: utf-8


import open3d as o3d
import numpy as np
import random
import os

def traverse_path_recursively(rootdir):
    filedirs = []
    def gci(filepath):
        files = os.listdir(filepath)
        for fi in files:
            fi_d = os.path.join(filepath,fi)            
            if os.path.isdir(fi_d):
                gci(fi_d)                  
            else:
                filedirs.append(os.path.join(filepath,fi_d))
        return
    gci(rootdir)
    return filedirs



def write_ply_data(filename, points, normals):
    '''
    write data to ply file.
    '''
    points = points.astype("int")
    normals = normals.astype("float")

    if os.path.exists(filename):
        os.system('rm '+filename)
    f = open(filename,'a+')
    #print('data.shape:',data.shape)
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(points.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.writelines(['property float nx\n','property float ny\n','property float nz\n'])
    f.write('end_header\n')
    for point, normal in zip(points, normals):
        f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), ' ',
                      str(round(normal[0],6)), ' ', str(round(normal[1],6)), ' ',str(round(normal[2],6)), '\n'])
    f.close() 

    return

def get_rotate_matrix():
    # random rotate
    m=np.eye(3,dtype='float32')
    m[0,0]*=np.random.randint(0,2)*2-1
    m=np.dot(m,np.linalg.qr(np.random.randn(3,3))[0])
    return m

def mesh2pc(mesh_filedir, pc_filedir, n_points=400000, resolution=255):
    # sample points uniformly.
    print(mesh_filedir)
    mesh = o3d.io.read_triangle_mesh(mesh_filedir)
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    except:
        print("RuntimeError", '!'*8)
        return
    # random rotate.
    points = np.asarray(pcd.points)
    points = np.dot(points, get_rotate_matrix())
    # voxelization.
    points = points - np.min(points)
    points = points / np.max(points)
    points = points * (resolution)
    points = np.round(points).astype('int')
    points = np.unique(points, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=20))

    print(pc_filedir, len(points))
    # o3d.io.write_point_cloud(pc_filedir, pcd, write_ascii=True)
    write_ply_data(pc_filedir, np.asarray(pcd.points), np.asarray(pcd.normals))

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_rootdir", type=str, default="/home/ubuntu/HardDisk1/ModelNet40", dest="input_rootdir")
    # /home/ubuntu/HardDisk1/ShapeNetCore.v2/
    parser.add_argument("--output_rootdir", type=str, default="../testdata/ModelNet40/", dest="output_rootdir")
    parser.add_argument("--n_testdata", type=int, default=32, dest="n_testdata")
    parser.add_argument("--n_points", type=int, default=400000, dest="n_points")
    parser.add_argument("--resolution", type=int, default=255, dest="resolution")
    args = parser.parse_args()
    print(args)


    filedirs = traverse_path_recursively(args.input_rootdir)
    print("input filedirs:\n", len(filedirs))
    off_filedirs = [f for f in filedirs if (os.path.splitext(f)[1]=='.off' or os.path.splitext(f)[1]=='.obj')]# .off or .obj
    filedirs = random.sample(off_filedirs, args.n_testdata)
    print("input filedirs:\n", len(filedirs))

    for idx, mesh_filedir in enumerate(filedirs):
        print(idx)
        pc_filedir = os.path.join(args.output_rootdir, str(idx) + '_' + mesh_filedir.split('/')[-1].split('.')[0] + '.ply')
        mesh2pc(mesh_filedir, pc_filedir, args.n_points, args.resolution)

    """
    python mesh2pc_open3d.py --n_testdata 100 --input_rootdir '/home/ubuntu/HardDisk1/ShapeNetCore.v2/' --output_rootdir '../testdata/ShapeNet/'
    """