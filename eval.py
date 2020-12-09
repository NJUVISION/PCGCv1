#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Nanjing University, Vision Lab.
# Last update: 
# 2019.10.27
# 2019.11.14
# 2020.11.26

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pylab  as plt
import pandas as pd
import subprocess
import glob
import configparser
import argparse
import importlib 
# from numba import cuda
tf.enable_eager_execution()

from process import preprocess, postprocess
# import models.model_voxception as model
from transform import compress_factorized, decompress_factorized
from transform import compress_hyper, decompress_hyper

from dataprocess.inout_bitstream import write_binary_files_factorized, read_binary_files_factorized
from dataprocess.inout_bitstream import write_binary_files_hyper, read_binary_files_hyper

os.environ['CUDA_VISIBLE_DEVICES']="0"
# set gpu.
cfg = tf.ConfigProto()
cfg.gpu_options.per_process_gpu_memory_fraction = 1.0
cfg.gpu_options.allow_growth = True
cfg.log_device_placement=True
# config.device_count={'gpu':0}
sess = tf.Session(config=cfg)

from myutils.pc_error_wrapper import pc_error
from myutils.pc_error_wrapper import get_points_number


def test_factorized(input_file, model, ckpt_dir, scale, cube_size, min_num, postfix=''):
    # Pre-process
    cubes, cube_positions, points_numbers = preprocess(input_file, scale, cube_size, min_num)
    ### Encoding
    strings, min_v, max_v, shape = compress_factorized(cubes, model, ckpt_dir)
    # Write files
    filename = os.path.split(input_file)[-1][:-4]
    print(filename)
    rootdir = './compressed'+ postfix +'/'
    bytes_strings, bytes_pointnums, bytes_cubepos = write_binary_files_factorized(
        filename, strings.numpy(), points_numbers, cube_positions,
        min_v.numpy(), max_v.numpy(), shape.numpy(), rootdir)
    # Read files
    strings_d, points_numbers_d, cube_positions_d, min_v_d, max_v_d, shape_d = \
        read_binary_files_factorized(filename, rootdir)
    # Decoding
    cubes_d = decompress_factorized(strings_d, min_v_d, max_v_d, shape_d, model, ckpt_dir)

    # bpp
    N = get_points_number(input_file)
    bpp = round(8*(bytes_strings + bytes_pointnums + bytes_cubepos)/float(N), 4)
    bpp_strings = round(8*bytes_strings/float(N), 4)
    bpp_pointsnums = round(8*bytes_pointnums/float(N) ,4)
    bpp_cubepos = round(8*bytes_cubepos/float(N), 4)
    bpp_strings_hyper = 0
    bpp_strings_head = 0
    bpps = [bpp, bpp_strings, bpp_strings_hyper, bpp_strings_head, bpp_pointsnums, bpp_cubepos]

    return cubes_d, cube_positions_d, points_numbers_d, N, bpps


def test_hyper(input_file, model, ckpt_dir, scale, cube_size, min_num, postfix=''):
    # Pre-process
    cubes, cube_positions, points_numbers = preprocess(input_file, scale, cube_size, min_num)
    ### Encoding
    y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape, x_ds =  compress_hyper(cubes, model, ckpt_dir, True)
    # Write files
    filename = os.path.split(input_file)[-1][:-4]
    print(filename)
    rootdir = './compressed'+ postfix +'/'
    bytes_strings, bytes_strings_head, bytes_strings_hyper, bytes_pointnums, bytes_cubepos = write_binary_files_hyper(
        filename, y_strings.numpy(), z_strings.numpy(), points_numbers, cube_positions, 
        y_min_vs.numpy(), y_max_vs.numpy(), y_shape.numpy(), 
        z_min_v.numpy(), z_max_v.numpy(), z_shape.numpy(), rootdir)
    # Read files
    y_strings_d, z_strings_d, points_numbers_d, cube_positions_d,  y_min_vs_d, y_max_vs_d, y_shape_d, z_min_v_d, z_max_v_d, z_shape_d =  \
        read_binary_files_hyper(filename, rootdir)
    # Decoding
    cubes_d = decompress_hyper(y_strings_d, y_min_vs_d.astype('int32'), y_max_vs_d.astype('int32'), 
                                y_shape_d, z_strings_d, z_min_v_d, z_max_v_d, z_shape_d, model, ckpt_dir)
    # cheat!!!
    ##############
    cubes_d = x_ds
    ##############
    # bpp
    N = get_points_number(input_file)
    bpp = round(8*(bytes_strings + bytes_strings_head + bytes_strings_hyper + 
                    bytes_pointnums + bytes_cubepos)/float(N), 4)

    bpp_strings = round(8*bytes_strings/float(N), 4)
    bpp_strings_hyper = round(8*bytes_strings_hyper/float(N), 4)
    bpp_strings_head = round(8*bytes_strings_head/float(N), 4)
    bpp_pointsnums = round(8*bytes_pointnums/float(N) ,4)
    bpp_cubepos = round(8*bytes_cubepos/float(N), 4)
    bpps = [bpp, bpp_strings, bpp_strings_hyper, bpp_strings_head, bpp_pointsnums, bpp_cubepos]

    return cubes_d, cube_positions_d, points_numbers_d, N, bpps


def collect_results(results, results_d1, results_d2, bpps, N, scale, rho_d1, rho_d2):
    # bpp
    results["ori_points"] = N
    results["scale"] = scale
    # results["cube_size"] = cube_size
    # results["res"] = res
    results["bpp"] = bpps[0]
    results["bpp_strings"] = bpps[1]
    results["bpp_strings_hyper"] = bpps[2]
    results["bpp_strings_head"] = bpps[3]
    results["bpp_pointsnums"] = bpps[4]
    results["bpp_cubepos"] = bpps[5]

    results["rho_d1"] = rho_d1
    results["optimal D1 PSNR"] = results_d1["mseF,PSNR (p2point)"]

    results["rho_d2"] = rho_d2
    results["optimal D2 PSNR"] = results_d2["mseF,PSNR (p2plane)"]

    print(results)

    return results


def plot_results(all_results, filename, root_dir):
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
            label="D1", marker='x', color='red')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
            label="D2", marker='x', color = 'blue')

    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["optimal D1 PSNR"][:]), 
            label="D1 (optimal)", marker='h', color='red', linestyle='-.')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["optimal D2 PSNR"][:]), 
            label="D2 (optimal)", marker='h', color='blue', linestyle='-.')
    plt.title(filename)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(root_dir, filename+'.png'))

    return 


def eval(input_file, rootdir, cfgdir, res, mode, cube_size, modelname, fixed_thres, postfix):
    # model = 'model_voxception'
    model = importlib.import_module(modelname)

    filename = os.path.split(input_file)[-1][:-4]
    output_file = filename + '_rec_' + postfix + '.ply'
    input_file_n = input_file    
    csv_rootdir = rootdir
    if not os.path.exists(csv_rootdir):
        os.makedirs(csv_rootdir)
    csv_name = os.path.join(csv_rootdir, filename + '.csv')

    config = configparser.ConfigParser()
    config.read(cfgdir)

    cube_size = config.getint('DEFAULT', 'cube_size')
    min_num = config.getint('DEFAULT', 'min_num')
    print('cube size:', cube_size, 'min num:', min_num, 'res:', res)

    for index, rate in enumerate(config.sections()):
        scale = float(config.get(rate, 'scale'))
        ckpt_dir = str(config.get(rate, 'ckpt_dir'))
        rho_d1 = float(config.get(rate, 'rho_d1'))
        rho_d2 = float(config.get(rate, 'rho_d2'))
        print('='*80, '\n', 'config:', rate, 'scale:', scale, 'ckpt_dir:', ckpt_dir, 'rho (d1):', rho_d1, 'rho_d2:', rho_d2)

        if mode=="factorized":
            cubes_d, cube_positions, points_numbers, N, bpps = test_factorized(input_file, model, ckpt_dir, scale, cube_size, min_num, postfix)
        elif mode == "hyper":
            cubes_d, cube_positions, points_numbers, N, bpps = test_hyper(input_file, model, ckpt_dir, scale, cube_size, min_num, postfix)
        cubes_d = cubes_d.numpy()
        print("bpp:",bpps[0])

        # metrics.
        rho = 1.0
        postprocess(output_file, cubes_d, points_numbers, cube_positions, scale, cube_size, rho, fixed_thres)
        results = pc_error(input_file, output_file, input_file_n, res, show=False)

        rho = rho_d1
        postprocess(output_file, cubes_d, points_numbers, cube_positions, scale, cube_size, rho, fixed_thres)
        results_d1 = pc_error(input_file, output_file, input_file_n, res, show=False)

        rho = rho_d2
        postprocess(output_file, cubes_d, points_numbers, cube_positions, scale, cube_size, rho, fixed_thres)
        results_d2 = pc_error(input_file, output_file, input_file_n, res, show=False)
         
        results = collect_results(results, results_d1, results_d2, bpps, N, scale, rho_d1, rho_d2)

        if index == 0:
            all_results = results.copy(deep=True)
        else:
            all_results = all_results.append(results, ignore_index=True)

        all_results.to_csv(csv_name, index=False)

    print(all_results)
    plot_results(all_results, filename, csv_rootdir)

    return all_results

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, nargs='+', default='', dest="input")
    parser.add_argument("--rootdir", type=str, default='results/hyper/', dest="rootdir")
    parser.add_argument("--cfgdir", type=str, default='results/hyper/8iVFB_vox10.ini', dest="cfgdir")
    parser.add_argument("--res", type=int, default=256, dest="res")
    parser.add_argument("--mode", type=str, default='hyper', dest="mode")

    parser.add_argument("--cube_size", type=int, default=64, dest="cube_size")
    parser.add_argument("--modelname", default="models.model_voxception", help="(model_simple, model_voxception)", dest="modelname")
    parser.add_argument("--fixed_thres", type=float, default=None, help="fixed threshold ", dest="fixed_thres")    
    parser.add_argument("--postfix", default="", help="", dest="postfix") 
    
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.rootdir):
        os.makedirs(args.rootdir)
        
    # shapenet_filedirs = glob.glob("testdata/ShapeNet/*.ply")
    # modelnet_filedirs = glob.glob("testdata/ModelNet40/*.ply")
    # args.input = modelnet_filedirs + shapenet_filedirs
    print(args.input)
    for input_file in sorted(args.input):
        print(input_file)
        all_results = eval(input_file, args.rootdir, args.cfgdir, args.res, args.mode, 
                            args.cube_size, args.modelname, args.fixed_thres, args.postfix)

    """
    python eval.py --input "testdata/8iVFB/longdress_vox10_1300.ply" \
                    --rootdir="results/hyper/" \
                    --cfgdir="results/hyper/8iVFB_vox10.ini" \
                    --res=1024
    """