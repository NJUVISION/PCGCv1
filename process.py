# Copyright (c) Nanjing University, Vision Lab.
# Last update:
# 2020.11.26 

import os
import numpy as np
import sys
import time
import random
# sys.path.append('.')
from dataprocess.inout_points import load_ply_data, write_ply_data, \
                                    load_points, save_points, \
                                    points2voxels, voxels2points, \
                                    select_voxels

def preprocess(input_file, scale, cube_size, min_num):
    """Scaling, Partition & Voxelization.
    Input: .ply file and arguments for pre-process.  
    Output: partitioned cubes, cube positions, and number of points in each cube. 
    """
    prefix=input_file.split('/')[-1].split('_')[0]+str(random.randint(1,100))
    print('===== Preprocess =====')
    # scaling (optional)
    start = time.time()
    if scale == 1:
        scaling_file = input_file 
    else:
        pc = load_ply_data(input_file)
        pc_down = np.round(pc.astype('float32') * scale)
        pc_down = np.unique(pc_down, axis=0)# remove duplicated points
        scaling_file = prefix+'downscaling.ply'
        write_ply_data(scaling_file, pc_down)
    print("Scaling: {}s".format(round(time.time()-start, 4)))

    # partition.
    start = time.time()
    partitioned_points, cube_positions = load_points(scaling_file, cube_size, min_num)
    print("Partition: {}s".format(round(time.time()-start, 4)))
    if scale != 1:
        os.system("rm "+scaling_file)

    # voxelization.
    start = time.time()
    cubes = points2voxels(partitioned_points, cube_size)
    points_numbers = np.sum(cubes, axis=(1,2,3,4)).astype(np.uint16)
    print("Voxelization: {}s".format(round(time.time()-start, 4)))

    print('cubes shape: {}'.format(cubes.shape))
    print('points numbers (sum/mean/max/min): {} {} {} {}'.format( 
    points_numbers.sum(), round(points_numbers.mean()), points_numbers.max(), points_numbers.min()))

    return cubes, cube_positions, points_numbers

def postprocess(output_file, cubes, points_numbers, cube_positions, scale, cube_size, rho, fixed_thres=None):
    """Classify voxels to occupied or free, then extract points and write to file.
    Input:  deocded cubes, cube positions, points numbers, cube size and rho=ouput numbers/input numbers.
    """
    prefix=output_file.split('/')[-1].split('_')[0]+str(random.randint(1,100))
    print('===== Post process =====')
    # Classify.
    start = time.time()
    output = select_voxels(cubes, points_numbers, rho, fixed_thres=fixed_thres)

    # Extract points.
    #points = voxels2points(output.numpy())
    points = voxels2points(output)
    print("Classify and extract points: {}s".format(round(time.time()-start, 4)))

    # scaling (optional)
    start = time.time()
    if scale == 1:
        save_points(points, cube_positions, output_file, cube_size)
    else:
        scaling_output_file = prefix+'downsampling_rec.ply'
        save_points(points, cube_positions, scaling_output_file, cube_size)
        pc = load_ply_data(scaling_output_file)
        pc_up = pc.astype('float32') * float(1/scale)
        write_ply_data(output_file, pc_up)
        os.system("rm "+scaling_output_file)
    print("Write point cloud to {}: {}s".format(output_file, round(time.time()-start, 4)))

    return

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input", type=str, default="testdata/8iVFB/longdress_vox10_1300.ply", dest="input")
    parser.add_argument(
        "--output", type=str, default="rec.ply", dest="output")
    parser.add_argument(
        "--scale", type=float, default=1.0, dest="scale")
    parser.add_argument(
        "--cube_size", type=int, default=64, dest="cube_size")
    parser.add_argument(
        "--min_num", type=int, default=20, dest="min_num")
    args = parser.parse_args()
    print(args)

    cubes, cube_positions, points_numbers = preprocess(args.input, 
                                                        args.scale, 
                                                        args.cube_size, 
                                                        args.min_num)
    print("===================\n", cubes.shape, np.sum(cubes), '\n', cube_positions.shape, '\n', points_numbers.shape)
    postprocess(args.output, cubes, points_numbers, cube_positions, args.scale, args.cube_size, rho=1.0)
    os.system("myutils/pc_error_d" \
    + ' -a ' + args.input + ' -b ' + args.output + " -r 1023")
    os.system("rm "+args.output)