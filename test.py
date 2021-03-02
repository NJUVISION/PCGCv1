# Copyright (c) Nanjing University, Vision Lab.
# Last update:
# 2020.11.26 

import os
import argparse
import numpy as np
import tensorflow as tf
import time
import importlib 
tf.enable_eager_execution()

from process import preprocess, postprocess

# import models.model_voxception as model

from transform import compress_factorized, decompress_factorized
from dataprocess.inout_bitstream import write_binary_files_factorized, read_binary_files_factorized

from transform import compress_hyper, decompress_hyper
from dataprocess.inout_bitstream import write_binary_files_hyper, read_binary_files_hyper


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("command", choices=["compress", "decompress"],
                        help="What to do: 'compress' reads a point cloud (.ply format) "
                            "and writes compressed binary files. 'decompress' "
                            "reads binary files and reconstructs the point cloud (.ply format). "
                            "input and output filenames need to be provided for the latter. ")
    parser.add_argument("input", nargs="?", help="Input filepath.")
    parser.add_argument("output", nargs="?", help="Output filepath.")
    parser.add_argument("--mode", type=str, default='hyper', dest="mode", help='factorized entropy model or hyper prior')
    parser.add_argument("--modelname", default="models.model_voxception", dest="modelname", help="(model_simple, model_voxception)")
    parser.add_argument("--ckpt_dir", type=str, default='', dest="ckpt_dir",  help='checkpoint')
    parser.add_argument("--scale", type=float, default=1.0, dest="scale", help="scaling factor.")
    parser.add_argument("--cube_size", type=int, default=64, dest="cube_size", help="size of partitioned cubes.")
    parser.add_argument("--min_num", type=int, default=64, dest="min_num", help="minimum number of points in a cube.")
    parser.add_argument("--rho", type=float, default=1.0, dest="rho", help="ratio of the numbers of output points to the number of input points.")
    parser.add_argument("--gpu", type=int, default=1, dest="gpu", help="use gpu (1) or not (0).") 
    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    """
    Examples:
    python test.py compress "testdata/8iVFB/longdress_vox10_1300.ply" \
        --ckpt_dir="checkpoints/factorized/a2b3/" --mode="factorized"
    python test.py decompress "compressed/longdress_vox10_1300" \
        --ckpt_dir="checkpoints/factorized/a2b3/" --mode="factorized"   

    python test.py compress "testdata/8iVFB/longdress_vox10_1300.ply" \
        --ckpt_dir="checkpoints/hyper/a6b3/" 
    python test.py decompress "compressed/longdress_vox10_1300" \
        --ckpt_dir="checkpoints/hyper/a6b3/"
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

    model = importlib.import_module(args.modelname)

    if args.mode == "factorized":
        if args.command == "compress":
            cubes, cube_positions, points_numbers = preprocess(args.input, args.scale, args.cube_size, args.min_num)
            strings, min_v, max_v, shape = compress_factorized(cubes, model, args.ckpt_dir)
            if not args.output:
                args.output = os.path.split(args.input)[-1][:-4]
                rootdir = './compressed'
            else:
                rootdir, args.output = os.path.split(args.output)
            bytes_strings, bytes_pointnums, bytes_cubepos = write_binary_files_factorized(
                args.output, strings.numpy(), points_numbers, cube_positions, min_v.numpy(), max_v.numpy(), shape.numpy(), rootdir=rootdir)

        elif args.command == "decompress":
            rootdir, filename = os.path.split(args.input)
            if not args.output:
                args.output = filename + "_rec.ply"
            strings_d, points_numbers_d, cube_positions_d, min_v_d, max_v_d, shape_d = read_binary_files_factorized(filename, rootdir)
            cubes_d = decompress_factorized(strings_d, min_v_d, max_v_d, shape_d, model, args.ckpt_dir)
            postprocess(args.output, cubes_d.numpy(), points_numbers_d, cube_positions_d, args.scale, args.cube_size, args.rho)
    
    if args.mode == "hyper":
        if args.command == "compress":
            if not args.output:
                args.output = os.path.split(args.input)[-1][:-4]
                rootdir = './compressed'
            else:
                rootdir, args.output = os.path.split(args.output)

            cubes, cube_positions, points_numbers = preprocess(args.input, args.scale, args.cube_size, args.min_num)
 
            y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape = compress_hyper(cubes, model, args.ckpt_dir)

            bytes_strings, bytes_strings_head, bytes_strings_hyper, bytes_pointnums, bytes_cubepos = write_binary_files_hyper(
                args.output, y_strings.numpy(), z_strings.numpy(), points_numbers, cube_positions,
                y_min_vs.numpy(), y_max_vs.numpy(), y_shape.numpy(), 
                z_min_v.numpy(), z_max_v.numpy(), z_shape.numpy(), rootdir=rootdir)

        elif args.command == "decompress":
            rootdir, filename = os.path.split(args.input)
            if not args.output:
                args.output = filename + "_rec.ply"

            y_strings_d, z_strings_d, points_numbers_d, cube_positions_d, \
            y_min_vs_d, y_max_vs_d, y_shape_d, z_min_v_d, z_max_v_d, z_shape_d = read_binary_files_hyper(filename, rootdir)

            cubes_d = decompress_hyper(y_strings_d, y_min_vs_d, y_max_vs_d, y_shape_d, z_strings_d, z_min_v_d, z_max_v_d, z_shape_d, model, args.ckpt_dir)
            
            postprocess(args.output, cubes_d.numpy(), points_numbers_d, cube_positions_d, args.scale, args.cube_size, args.rho)
            
