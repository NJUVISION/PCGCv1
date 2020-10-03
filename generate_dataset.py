# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Hao Zhu, Zhan Ma, Tong Chen, Haojie Liu, Qiu Shen; Nanjing University, Vision Lab.
# Chaofei Wang; Shanghai Jiao Tong University, Cooperative Medianet Innovation Center.
# Last update: 2019.09.17

import numpy as np
import h5py
import os
import glob
import random
from dataprocess.inout_points import load_points

def generate_dataset(INPUT_DIR, OUTPUT_DIR, DATA_NUM, cube_size=64):
    # read file
    plydirs = glob.glob(INPUT_DIR + '*.ply')
    random.shuffle(plydirs)
    random.shuffle(plydirs)
    print('ply file direction list length:', len(plydirs))

    num_data = 0
    for filename in plydirs:
        set_points, _ = load_points(filename,  cube_size=cube_size, min_num=20)
        random.shuffle(set_points)
        # only half of the points will be used
        for i, points in enumerate(set_points):
            points = points.astype("uint8")
            points_dir = os.path.join(OUTPUT_DIR, 
            filename.split('/')[-1].split('.')[0]+'_'+str(i)+'n.h5')
            with h5py.File(points_dir, 'w') as h:
                # print(set_points[i].dtype)
                h.create_dataset('data', data=points, shape=points.shape)
            num_data += 1

            if num_data%100==0:
                print(num_data)
                
        if num_data > DATA_NUM:
            break
    
    return


if __name__ == "__main__":
    VOXEL_SIZE = 64
    DATA_DIR = '/home/ubuntu/HardDisk1/geometry_training_datasets/shapenet_points255/shapenet_points255_part2/'
    print('data direction:', DATA_DIR)
    OUTPUT_DIR ='/home/ubuntu/HardDisk1/points'+str(VOXEL_SIZE)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    DATA_NUM = 1e6
    generate_dataset(DATA_DIR, OUTPUT_DIR, DATA_NUM, VOXEL_SIZE)
