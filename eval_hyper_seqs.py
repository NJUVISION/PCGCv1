#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2019.10.27
# 2019.11.14


# # Evaluate R-D performance

# In[2]:


import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pylab  as plt
import pandas as pd
import subprocess
import glob
import configparser
from numba import cuda
tf.enable_eager_execution()


# In[3]:


from mycodec_hyper import preprocess, postprocess, compress, decompress, write_binary_files, read_binary_files

# if necessary, maybe you should consider compression with less memory, which eocode each block respectively.
from mycodec_hyper import preprocess, postprocess, compress_less_mem, decompress_less_mem, write_binary_files_less_mem, read_binary_files_less_mem


# In[4]:


import models.model_voxception as model


# In[5]:


os.environ['CUDA_VISIBLE_DEVICES']="0"

# set gpu.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
config.log_device_placement=True
# config.device_count={'gpu':0}
sess = tf.Session(config=config)


# In[6]:


from myutils.pc_error_wrapper import pc_error
from myutils.pc_error_wrapper import get_points_number


# ## Set config

# In[16]:


def set_config(input_file, resolution):
    filename = os.path.split(input_file)[-1][:-4]
    config_file = os.path.join('./config/hyper/', filename+'.ini')
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
        print('config already exists.')
    else:
        config["DEFAULT"] = {"cube_size": 64, 
                            "min_num": 64,
                            "resolution":resolution}

        config["R1"] = {"scale": 5/8.,
                       "ckpt_dir": './checkpoints/hyper/a0.75b3/'}

        config["R2"] = {"scale": 1.0,
                       "ckpt_dir": './checkpoints/hyper/a0.75b3/'}

        config["R3"] = {"scale": 1.0,
                       "ckpt_dir": './checkpoints/hyper/a2b3/'}

        config["R4"] = {"scale": 1.0,
                       "ckpt_dir": './checkpoints/hyper/a3.5b3/'}

        config["R5"] = {"scale": 1.0,
                       "ckpt_dir": './checkpoints/hyper/a6b3/'}

        config["R6"] = {"scale": 1.0,
                       "ckpt_dir": './checkpoints/hyper/a10b3/'}

        config["R7"] = {"scale": 1.0,
                       "ckpt_dir": './checkpoints/hyper/a16b3/'}
        config.write(open(config_file, 'w'))

        print('initialize config.')
        
    return config, config_file


# ### Select rho

# In[17]:


def select_rho(item, input_file, output_file, input_file_n, 
               cubes_d, points_numbers_d, cube_positions_d, scale, cube_size, res):

    steps = [0.02]*4+[0.04]*4+[0.08]*4+[0.16]*4+[0.32]*4
    MAX = 0
    rho = 1.0
    optimal_rho = 1.0

    for i, step in enumerate(steps):
        print('===== select rho =====')
        postprocess(output_file, cubes_d, points_numbers_d, cube_positions_d, scale, cube_size, rho)
        results = pc_error(input_file, output_file, input_file_n, res, show=False)
        """
        # record results.
        results["n_points"] = get_points_number(output_file)
        results["rho"] = rho
        if i == 0:
            all_results = results.copy(deep=True)
        else:
            all_results = all_results.append(results, ignore_index=True)
        """

        PSNR = float(results[item])
        print('===== results: ', i, rho, item, PSNR)

        MAX = max(PSNR, MAX)
        if PSNR < MAX:
            break
        else:
            optimal_rho = rho

        if item == "mseF,PSNR (p2point)":
            rho += step
        elif item == "mseF,PSNR (p2plane)":
            rho -= step
        else:
            print('ERROR', item)
            break
    
    return optimal_rho


# ## test

# In[18]:


def eval(input_file, config, config_file):
    # model = 'model_voxception'
    filename = os.path.split(input_file)[-1][:-4]
    output_file = filename + '_rec.ply'
    # input_file_n = input_file[:-4]+'_n.ply'
    input_file_n = input_file    
    
    cube_size = config.getint('DEFAULT', 'cube_size')
    min_num = config.getint('DEFAULT', 'min_num')
    res = config.getint('DEFAULT', 'resolution')

    print('cube size:', cube_size, 'min num:', min_num, 'res:', res)

    for index, rate in enumerate(config.sections()):
        scale = float(config.get(rate, 'scale'))
        ckpt_dir = str(config.get(rate, 'ckpt_dir'))
        print('====================', 'config:', rate, 'scale:', scale, 'ckpt_dir:', ckpt_dir)

        # Pre-process
        cubes, cube_positions, points_numbers = preprocess(input_file, scale, cube_size, min_num)
        ### Encoding
        y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape, x_ds =         compress_less_mem(cubes, model, ckpt_dir, True)
        # Write files
        filename = os.path.split(input_file)[-1][:-4]
        print(filename)
        rootdir = './compressed/'
        bytes_strings, bytes_strings_head, bytes_strings_hyper, bytes_pointnums, bytes_cubepos =         write_binary_files_less_mem(
        filename, y_strings.numpy(), z_strings.numpy(), points_numbers, cube_positions,
        y_min_vs.numpy(), y_max_vs.numpy(), y_shape.numpy(), 
        z_min_v.numpy(), z_max_v.numpy(), z_shape.numpy(), rootdir)
        # Read files
        y_strings_d, z_strings_d, points_numbers_d, cube_positions_d,         y_min_vs_d, y_max_vs_d, y_shape_d, z_min_v_d, z_max_v_d, z_shape_d =         read_binary_files_less_mem(filename, rootdir)
        # Decoding
        cubes_d = decompress_less_mem(y_strings_d, y_min_vs_d.astype('int32'), y_max_vs_d.astype('int32'), 
                                      y_shape_d, z_strings_d, z_min_v_d, z_max_v_d, z_shape_d, model, ckpt_dir)
        # cheat!!!
        ##############
        cubes_d = x_ds
        ##############
        # bpp
        N = get_points_number(input_file)
        bpp = round(8*(bytes_strings + 
                       bytes_strings_head + 
                       bytes_strings_hyper + 
                       bytes_pointnums + 
                       bytes_cubepos)/float(N), 4)

        bpp_strings = round(8*bytes_strings/float(N), 4)
        bpp_strings_head = round(8*bytes_strings_head/float(N), 4)
        bpp_strings_hyper = round(8*bytes_strings_hyper/float(N), 4)
        bpp_pointsnums = round(8*bytes_pointnums/float(N) ,4)
        bpp_cubepos = round(8*bytes_cubepos/float(N), 4)

        ########## Post-process ##########
        # select rho for optimal d1/d2 metrics.
        if config.has_option(rate, 'rho_d1'):
            rho_d1 = float(config.get(rate, 'rho_d1'))
        else:
            rho_d1 = select_rho("mseF,PSNR (p2point)", input_file, output_file, input_file_n, 
                            cubes_d, points_numbers_d, cube_positions_d, scale, cube_size, res)
            config.set(rate, 'rho_d1', str(rho_d1)) 
            config.write(open(config_file, 'w'))

        if config.has_option(rate, 'rho_d2'):
            rho_d2 = float(config.get(rate, 'rho_d2'))
        else:
            rho_d2 = select_rho("mseF,PSNR (p2plane)", input_file, output_file, input_file_n, 
                            cubes_d, points_numbers_d, cube_positions_d, scale, cube_size, res)
            config.set(rate, 'rho_d2', str(rho_d2))
            config.write(open(config_file, 'w'))

        # metrics.
        for index_rho, rho in enumerate((1.0, rho_d1, rho_d2)):
            postprocess(output_file, cubes_d, points_numbers_d, cube_positions_d, scale, cube_size, rho)

            # distortion
            results = pc_error(input_file, output_file, input_file_n, res, show=False)

            # bpp
            results["n_points"] = get_points_number(output_file)
            results["rho"] = rho
            results["ori_points"] = N
            results["scale"] = scale
            results["bpp_strings"] = bpp_strings
            results["bpp_strings_head"] = bpp_strings_head
            results["bpp_strings_hyper"] = bpp_strings_hyper
            results["bpp_pointsnums"] = bpp_pointsnums
            results["bpp_cubepos"] = bpp_cubepos
            results["bpp"] = bpp

            print(results)

            if index_rho == 0:
                if index == 0:
                    all_results = results.copy(deep=True)
                else:
                    all_results = all_results.append(results, ignore_index=True)
            elif index_rho == 1:
                if index == 0:
                    all_results_d1 = results.copy(deep=True)
                else:
                    all_results_d1 = all_results_d1.append(results, ignore_index=True)
            else:
                if index == 0:
                    all_results_d2 = results.copy(deep=True)
                else:
                    all_results_d2 = all_results_d2.append(results, ignore_index=True)    

    # write to csv
    print(all_results)
    print(all_results_d1)
    print(all_results_d2)
    csv_root_dir = './CSV/hyper/'
    if not os.path.exists(csv_root_dir):
        os.makedirs(csv_root_dir)

    csv_name = os.path.join(csv_root_dir, filename + '.csv')
    all_results.to_csv(csv_name, index=False)

    csv_name_d1 = os.path.join(csv_root_dir, filename + '_d1.csv')
    all_results_d1.to_csv(csv_name_d1, index=False)

    csv_name_d2 = os.path.join(csv_root_dir, filename + '_d2.csv')
    all_results_d2.to_csv(csv_name_d2, index=False)
    
    return all_results, all_results_d1, all_results_d2


# ### Plot performance

# In[19]:


def plot(input_file, all_results, all_results_d1, all_results_d2):
    fig, ax = plt.subplots(figsize=(7.3, 4.2))

    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
            label="D1", marker='x', color='red')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
            label="D2", marker='x', color = 'blue')

    plt.plot(np.array(all_results_d1["bpp"][:]), np.array(all_results_d1["mseF,PSNR (p2point)"][:]), 
            label="D1 (optimal)", marker='h', color='red', linestyle='-.')
    plt.plot(np.array(all_results_d2["bpp"][:]), np.array(all_results_d2["mseF,PSNR (p2plane)"][:]), 
            label="D2 (optimal)", marker='h', color='blue', linestyle='-.')

    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["h.,PSNR   (p2point)"][:]), 
            label="D1 Hausdorff", marker='x', color = 'green')
   
    csv_root_dir = './CSV/hyper/'
    filename = os.path.split(input_file)[-1][:-4]

    plt.title(filename)
    plt.xlabel('bpp')
    plt.ylabel('RSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    #filename = os.path.split(input_file)[-1][:-4]
    fig.savefig(os.path.join(csv_root_dir, filename+'.png'))
    
    return 


# # run

# In[20]:


# In[ ]:


eightiVFB = sorted(glob.glob('testdata/8iVFB/'+'*.ply'))
MVUB = sorted(glob.glob('testdata/MVUB/'+'*.ply'))
owlii = sorted(glob.glob('testdata/Owlii/'+'*.ply'))
# inanimate = sorted(glob.glob('testdata/GPCC_inanimate/'+'*.ply'))

# In[ ]:


for input_file in eightiVFB:
    config, config_file = set_config(input_file, resolution=1024)
    all_results, all_results_d1, all_results_d2 = eval(input_file, config, config_file)
    plot(all_results, all_results_d1, all_results_d2)


# In[ ]:


for input_file in MVUB:
    print(input_file)
    config, config_file = set_config(input_file, resolution=512)
    all_results, all_results_d1, all_results_d2 = eval(input_file, config, config_file)
    plot(input_file, all_results, all_results_d1, all_results_d2)


# In[ ]:


for input_file in owlii:
    print(input_file)
    config, config_file = set_config(input_file, resolution=2048)
    all_results, all_results_d1, all_results_d2 = eval(input_file, config, config_file)
    plot(input_file, all_results, all_results_d1, all_results_d2)







