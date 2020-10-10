# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2019.10.08

import os
import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import h5py
import argparse
import importlib 
import time
import glob
import random
random.seed(3)

from models.entropy_model import EntropyBottleneck
from dataprocess.inout_points import load_points, save_points, points2voxels
from loss import get_bce_loss, get_classify_metrics
from dataprocess.post_process import select_voxels
import models.model_voxception as model
# set gpu.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument(
#     "--model", default="model_voxception",
#     help="model name.")
parser.add_argument(
    "--alpha", type=float, default=2., dest="alpha",
    help="weights for distortion.")
parser.add_argument(
    "--beta", type=float, default=3., dest="beta",
    help="Weight for empty position.")
parser.add_argument(
    "--prefix", type=str, default='', dest="prefix",
    help="prefix of checkpoints foloder.")
parser.add_argument(
  "--init_ckpt_dir", type=str, default='', dest="init_ckpt_dir",
  help='initial checkpoint direction.')
parser.add_argument(
  "--reset_optimizer", type=int, default=0, dest="reset_optimizer",
  help='reset optimizer (1) or not.')
parser.add_argument(
  "--batch_size", type=int, default=4, dest="batch_size",
  help='batch_size')
args = parser.parse_args()
print(args)

# model = importlib.import_module(args.model)

# Define parameters.
BATCH_SIZE = args.batch_size
DISPLAY_STEP = 10
SAVE_STEP = 4000
RATIO_EVAL = 10 #
NUM_ITEATION = 3e5 
alpha = args.alpha
beta = args.beta
init_ckpt_dir = args.init_ckpt_dir


# Define variables.
analysis_transform, synthesis_transform = model.AnalysisTransform(), model.SynthesisTransform()
entropy_bottleneck = EntropyBottleneck()

global_step = tf.train.get_or_create_global_step()
lr = 1e-05
main_optimizer = tf.train.AdamOptimizer(learning_rate = lr)

## Define checkpoint. 
if args.reset_optimizer == 0:
  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform,
                              synthesis_transform=synthesis_transform,
                              estimator=entropy_bottleneck,
                              global_step=global_step)
else:
  checkpoint = tf.train.Checkpoint(main_optimizer=main_optimizer,
                                analysis_transform=analysis_transform,
                                synthesis_transform=synthesis_transform,
                                estimator=entropy_bottleneck,
                                global_step=global_step)



file_list = sorted(glob.glob('/home/ubuntu/HardDisk1/geometry_training_datasets/points64/points64_part1/'+'*.h5'))
print('numbers of training data: ', len(file_list))

def eval(data, batch_size):
  bpps = 0.
  IoUs = 0. 
  # generate input data.
  for i in range(len(data)//batch_size):
    samples = data[i*batch_size:(i+1)*batch_size]
    samples_points = []
    for _, f in enumerate(samples):
      points = h5py.File(f, 'r')['data'][:].astype('int')
      samples_points.append(points)
    voxels = points2voxels(samples_points, 64).astype('float32')

    x = tf.convert_to_tensor(voxels)
    y = analysis_transform(x)
    y_tilde, likelihoods = entropy_bottleneck(y, training=False)# TODO: repalce noise by quantization.
    x_tilde = synthesis_transform(y_tilde)

    num_points = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(x, -1), 0), 'float32'))
    train_bpp_ae = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_points)

    points_nums = tf.cast(tf.reduce_sum(x, axis=(1,2,3,4)), 'int32')
    output = select_voxels(x_tilde, points_nums, 1.0)
    output = output.numpy()
    _, _, IoU = get_classify_metrics(output, x)

    bpps = bpps + train_bpp_ae
    IoUs = IoUs + IoU

  return bpps/(i+1), IoUs/(i+1)

def train():
  start = time.time()
  train_list = file_list[len(file_list)//RATIO_EVAL:]

  train_bpp_ae_sum = 0.
  train_IoU_sum = 0.
  num = 0.

  for step in range(int(global_step), int(NUM_ITEATION+1)):
    # generate input data.
    samples = random.sample(train_list, BATCH_SIZE)
    samples_points = []
    for _, f in enumerate(samples):
      points = h5py.File(f, 'r')['data'][:].astype('int')
      samples_points.append(points)
    voxels = points2voxels(samples_points, 64).astype('float32')
    x = tf.convert_to_tensor(voxels)
    
    with tf.GradientTape() as model_tape:
      y = analysis_transform(x)
      y_tilde, likelihoods = entropy_bottleneck(y, training=True)
      x_tilde = synthesis_transform(y_tilde)

      # losses.
      num_points = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(x, -1), 0), 'float32'))
      train_bpp_ae = tf.reduce_sum(tf.log(likelihoods)) / -np.log(2) / num_points
      train_zeros, train_ones = get_bce_loss(x_tilde, x)
      train_distortion = beta * train_zeros + 1.0 * train_ones
      train_loss = alpha * train_distortion +  1.0 * train_bpp_ae
      # metrics.
      _, _, IoU = get_classify_metrics(x_tilde, x)
  
      # gradients.
      gradients = model_tape.gradient(train_loss,
                                      analysis_transform.variables +
                                      synthesis_transform.variables +
                                      entropy_bottleneck.variables)
      # optimization.
      main_optimizer.apply_gradients(zip(gradients, 
                                          analysis_transform.variables +
                                          synthesis_transform.variables +
                                          entropy_bottleneck.variables))

    # post-process: classification.
    points_nums = tf.cast(tf.reduce_sum(x, axis=(1,2,3,4)), 'int32')
    output = select_voxels(x_tilde, points_nums, 1.0)
    output = output.numpy()  

    train_bpp_ae_sum += train_bpp_ae
    _, _, IoU = get_classify_metrics(output, x)
    train_IoU_sum += IoU
    num += 1

    # Display.
    if (step + 1) % DISPLAY_STEP == 0:
      train_bpp_ae_sum /= num
      train_IoU_sum  /= num

      print("Iteration:{0:}".format(step))
      print("Bpps: {0:.4f}".format(train_bpp_ae_sum.numpy()))
      print("IoU: ", train_IoU_sum.numpy())
      print('Running time:(mins):', round((time.time()-start)/60.))
      print()

      with writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
        tf.contrib.summary.scalar('bpp', train_bpp_ae_sum)
        tf.contrib.summary.scalar('IoU',train_IoU_sum)

      num = 0.
      train_bpp_ae_sum = 0.
      train_bpp_hyper_sum = 0.
      train_IoU_sum = 0.

      print('evaluating...')
      eval_list =random.sample(file_list[:len(file_list)//RATIO_EVAL], 16)
      bpp_eval, IoU_eval = eval(eval_list, batch_size=8)
      print("BPP:{0:.4f}, IoU:{1:.4f}".format(bpp_eval, IoU_eval))
      with eval_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('bpp', bpp_eval)
        tf.contrib.summary.scalar('IoU', IoU_eval)  

    # Update global steps.
    global_step.assign_add(1)
   
    # Save checkpoints.
    if (step + 1) % SAVE_STEP == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

# checkpoint.
checkpoint_dir = os.path.join('./checkpoints', \
  args.prefix+'a{0:.2f}b{1:.2f}/'.format(alpha, beta))
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
  status = checkpoint.restore(ckpt.model_checkpoint_path)
  print('Loading checkpoints...')
elif init_ckpt_dir !='':
  init_ckpt = tf.train.latest_checkpoint(checkpoint_dir=init_ckpt_dir)
  print('init_ckpt:', init_ckpt)
  status = checkpoint.restore(init_ckpt)
  global_step.assign(0)
  print('Loading initial checkpoints from {}...'.format(init_ckpt_dir))

log_dir = os.path.join('./logs', \
  args.prefix+'a{0:.2f}b{1:.2f}/'.format(alpha, beta))

eval_log_dir = os.path.join('./logs', \
  args.prefix+'eval_a{0:.2f}b{1:.2f}/'.format(alpha, beta))

writer = tf.contrib.summary.create_file_writer(log_dir)
eval_writer = tf.contrib.summary.create_file_writer(eval_log_dir)

if __name__ == "__main__":
  train()
