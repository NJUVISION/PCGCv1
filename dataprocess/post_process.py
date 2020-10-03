# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Hao Zhu, Zhan Ma, Tong Chen, Haojie Liu, Qiu Shen; Nanjing University, Vision Lab.
# Chaofei Wang; Shanghai Jiao Tong University, Cooperative Medianet Innovation Center.
# Last update: 2019.10.04

import tensorflow as tf
import numpy as np
import h5py
tf.enable_eager_execution()

# def select_voxels(vols, points_nums, offset_ratio=1.0, init_thres=-1.0):
#     '''Select the top k voxels and generate the mask.
#     input:  vols: [batch_size, vsize, vsize, vsize, 1] float32
#             points numbers: [batch_size]
#     output: the mask (0 or 1) representing the selected voxels: [batch_size, vsize, vsize, vsize]  
#     '''
#     vols = tf.squeeze(tf.convert_to_tensor(vols, dtype='float32'), axis=-1)
#     points_nums = tf.cast(tf.convert_to_tensor(points_nums), 'float32')
#     offset_ratio = tf.convert_to_tensor(offset_ratio, dtype='float32')

#     masks = vols[0:1]# just a place holder.

#     for i in range(vols.shape[0]):
#         vol = tf.gather(vols, i, axis=0)
#         num = tf.cast(offset_ratio* points_nums[i], 'int32')
#         thres = get_thres(vol, num, init_thres)
#         mask = tf.cast(tf.greater(vol, thres), 'float32')
#         masks = tf.concat([masks, tf.expand_dims(mask, 0)], axis=0)

#     return tf.expand_dims(masks[1:], -1)

# def get_thres(vol, num, init_thres):

#     # filter out most values by the initial threshold.
#     values = tf.gather_nd(vol, tf.where(vol > init_thres))
#     # number of values should be larger than expected number.
#     if tf.shape(values)[0] < num:
#         values = tf.reshape(vol, [-1])

#     # only sort the selected values.
#     sorted_values, _ = tf.nn.top_k(values, num)
#     thres = sorted_values[-1]

#     return thres


def select_voxels(vols, points_nums, offset_ratio=1.0, init_thres=-2.0):
    '''Select the top k voxels and generate the mask.
    input:  vols: [batch_size, vsize, vsize, vsize, 1] float32
            points numbers: [batch_size]
    output: the mask (0 or 1) representing the selected voxels: [batch_size, vsize, vsize, vsize]  
    '''
    vols = vols.numpy()
    points_nums = points_nums
    offset_ratio = offset_ratio

    masks = []

    for idx, vol in enumerate(vols):
        num = int(offset_ratio* points_nums[idx])
        thres = get_thres(vol, num, init_thres)
        mask = np.greater(vol, thres).astype('float32')

        masks.append(mask)

    return np.stack(masks)

def get_thres(vol, num, init_thres):

    # filter out most values by the initial threshold.
    #values = np.gather_nd(vol, np.where(vol > init_thres))
    values = vol[vol>init_thres]
    # number of values should be larger than expected number.
    if values.shape[0] < num:
        values = np.reshape(vol, [-1])

    # only sort the selected values.
    values.sort()

    thres = values[-num]
    #sorted_values, _ = tf.nn.top_k(values, num)
    #thres = sorted_values[-1]

    return thres


if __name__=='__main__':
    # set gpu.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    config.log_device_placement=True
    # config.device_count={'gpu':0}
    sess = tf.Session(config=config)

    data = np.random.rand(4, 64, 64, 64, 1) * (100) -50
    data = tf.convert_to_tensor(data, 'float32')
    points_nums = np.array([1000, 200, 10000, 50])
    offset_ratio = 1.0 
    init_thres = -1.0

    mask = select_voxels(data, points_nums, offset_ratio, init_thres)   
    print(mask)
    
