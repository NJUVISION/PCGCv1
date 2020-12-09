# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2019.10.08
"""
python train_hyper.py --alpha=0.5 --beta=3 --prefix='hyper_' --batch_size=1 \
  --init_ckpt_dir='checkpoints/hyper/a0.75b3' --reset_optimizer=1
"""

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
from models.conditional_entropy_model import SymmetricConditional
from dataprocess.inout_points import load_points, save_points, points2voxels, select_voxels
from loss import get_bce_loss, get_classify_metrics
import models.model_voxception as model
# set gpu.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# def parameters
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument(
#     "--model", default="model_voxception",
#     help="model.")
parser.add_argument(
    "--alpha", type=float, default=2., dest="alpha",
    help="weights for distoration.")
parser.add_argument(
    "--beta", type=float, default=3., dest="beta",
    help="Weight for empty position.")
parser.add_argument(
    "--gamma", type=float, default=1., dest="gamma",
    help="Weight for hyper likelihoods.")
parser.add_argument(
    "--delta", type=float, default=1., dest="delta",
    help="Weight for latent likelihoods.")
parser.add_argument(
    "--lr", type=float, default=1e-5, dest="lr",
    help="learning rate.")
parser.add_argument(
    "--num_iteration", type=int, default=3e5, dest="num_iteration",
    help="number of iteration.")
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
    "--lower_bound", type=float, default=1e-9, dest="lower_bound",
    help="lower bound of scale. 1e-5 or 1e-9")
parser.add_argument(
  "--batch_size", type=int, default=8, dest="batch_size",
  help='batch_size')
args = parser.parse_args()
print(args)

# model = importlib.import_module(args.model)

# Define parameters.
BATCH_SIZE = args.batch_size
DISPLAY_STEP = 100
SAVE_STEP = 5000
RATIO_EVAL = 9 #
# NUM_ITEATION = 2e5 
NUM_ITEATION = int(args.num_iteration)
alpha = args.alpha
beta = args.beta
gamma = args.gamma # weight of hyper prior.
delta = args.delta # weight of latent representation.
init_ckpt_dir = args.init_ckpt_dir
lower_bound = args.lower_bound
print('lower bound of scale:', lower_bound)
reset_optimizer = bool(args.reset_optimizer)
print('reset_optimizer:::', reset_optimizer)

# Define variables
analysis_transform, synthesis_transform = model.AnalysisTransform(), model.SynthesisTransform()
hyper_encoder, hyper_decoder = model.HyperEncoder(), model.HyperDecoder()

entropy_bottleneck = EntropyBottleneck()
conditional_entropy_model = SymmetricConditional()

global_step = tf.train.get_or_create_global_step()

# lr = tf.train.exponential_decay(1e-4, global_step, 20000, 0.75, staircase=True)
# lr = 1e-5
lr = args.lr
main_optimizer = tf.train.AdamOptimizer(learning_rate = lr)

########## Define checkpoint ########## 
if args.reset_optimizer == 0:
  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform,
                              synthesis_transform=synthesis_transform,
                              hyper_encoder = hyper_encoder,
                              hyper_decoder = hyper_decoder,
                              estimator=entropy_bottleneck,
                              global_step=global_step)
else:
  checkpoint = tf.train.Checkpoint(main_optimizer=main_optimizer,
                                analysis_transform=analysis_transform,
                                synthesis_transform=synthesis_transform,
                                hyper_encoder = hyper_encoder,
                                hyper_decoder = hyper_decoder,
                                estimator=entropy_bottleneck,
                                global_step=global_step)

file_list = glob.glob('/home/ubuntu/HardDisk1/geometry_training_datasets/points64/points64_part1/'+'*.h5')
print('numbers of training data: ', len(file_list))

def eval(data, batch_size):
  bpps_ae = 0.
  bpps_hyper = 0.
  IoUs = 0. 

  for i in range(len(data)//batch_size):
    samples = data[i*batch_size:(i+1)*batch_size]
    samples_points = []
    for _, f in enumerate(samples):
      points = h5py.File(f, 'r')['data'][:].astype('int')
      samples_points.append(points)
    voxels = points2voxels(samples_points, 64).astype('float32')

    x = tf.convert_to_tensor(voxels)
    y = analysis_transform(x)
    z = hyper_encoder(y)
    z_tilde, likelihoods_hyper = entropy_bottleneck(z, training=False)
    loc, scale = hyper_decoder(z_tilde)
    scale = tf.maximum(scale, lower_bound)
    y_tilde, likelihoods = conditional_entropy_model(y, loc, scale, training=False)
    x_tilde = synthesis_transform(y_tilde)

    num_points = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(x, -1), 0), 'float32'))
    train_bpp_ae = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_points)
    train_bpp_hyper = tf.reduce_sum(tf.log(likelihoods_hyper)) / (-np.log(2) * num_points)

    points_nums = tf.cast(tf.reduce_sum(x, axis=(1,2,3,4)), 'int32')
    x_tilde = x_tilde.numpy()
    output = select_voxels(x_tilde, points_nums, 1.0)
    # output = output.numpy()
    _, _, IoU = get_classify_metrics(output, x)

    bpps_ae = bpps_ae + train_bpp_ae
    bpps_hyper = bpps_hyper + train_bpp_hyper
    IoUs = IoUs + IoU

  return bpps_ae/(i+1), bpps_hyper/(i+1), IoUs/(i+1)


def train():
  start = time.time()
  train_list = file_list[len(file_list)//RATIO_EVAL:]

  train_bpp_ae_sum = 0.
  train_bpp_hyper_sum = 0.
  train_IoU_sum = 0.
  num = 0.

  for step in range(int(global_step), int(NUM_ITEATION+1)):
    # generate input data
    samples = random.sample(train_list, BATCH_SIZE)
    samples_points = []
    for _, f in enumerate(samples):
      points = h5py.File(f, 'r')['data'][:].astype('int')
      samples_points.append(points)
    voxels = points2voxels(samples_points, 64).astype('float32')
    x = tf.convert_to_tensor(voxels)
    
    with tf.GradientTape() as model_tape:
      y = analysis_transform(x)
      z = hyper_encoder(y)
      z_tilde, likelihoods_hyper = entropy_bottleneck(z, training=True)
      loc, scale = hyper_decoder(z_tilde)
      scale = tf.maximum(scale, lower_bound)# start with large lower bound to avaid crashes!
      y_tilde, likelihoods = conditional_entropy_model(y, loc, scale, training=True)
      x_tilde = synthesis_transform(y_tilde)

      # losses.
      num_points = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(x, -1), 0), 'float32')) 
      train_bpp_ae = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_points)
      train_bpp_hyper = tf.reduce_sum(tf.log(likelihoods_hyper)) / (-np.log(2) * num_points)
      train_zeros, train_ones = get_bce_loss(x_tilde, x)
      train_distortion = beta * train_zeros + 1.0 * train_ones
      train_loss = alpha * train_distortion +  delta * train_bpp_ae + gamma * train_bpp_hyper

      #gradients.
      gradients = model_tape.gradient(train_loss, 
                                      analysis_transform.variables + 
                                      synthesis_transform.variables +
                                      hyper_encoder.variables +
                                      hyper_decoder.variables +
                                      entropy_bottleneck.variables)
      # optimization.
      main_optimizer.apply_gradients(zip(gradients, 
                                        analysis_transform.variables + 
                                        synthesis_transform.variables +
                                        hyper_encoder.variables +
                                        hyper_decoder.variables +
                                        entropy_bottleneck.variables))

    # post-process: classification.
    points_nums = tf.cast(tf.reduce_sum(x, axis=(1,2,3,4)), 'int32')
    x_tilde = x_tilde.numpy()
    output = select_voxels(x_tilde, points_nums, 1.0)
    # output = output.numpy()  

    train_bpp_ae_sum += train_bpp_ae
    train_bpp_hyper_sum += train_bpp_hyper
    _, _, IoU = get_classify_metrics(output, x)
    train_IoU_sum += IoU
    num += 1

    # Display.
    if (step + 1) % DISPLAY_STEP == 0:
      train_bpp_ae_sum /= num
      train_bpp_hyper_sum /= num
      train_IoU_sum  /= num

      print("Iteration:{0:}".format(step))
      print("Bpps: {0:.4f} + {1:.4f}".format(train_bpp_ae_sum, train_bpp_hyper_sum))
      print("IoU: ", train_IoU_sum.numpy())
      print('Running time:(mins):', round((time.time()-start)/60.))
      print()

      with writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('bpp_ae', train_bpp_ae_sum)
        tf.contrib.summary.scalar('bpp_hyper', train_bpp_hyper_sum)
        tf.contrib.summary.scalar('bpp', train_bpp_ae_sum + train_bpp_hyper_sum)
        tf.contrib.summary.scalar('IoU',train_IoU_sum)
      
      num = 0.
      train_bpp_ae_sum = 0.
      train_bpp_hyper_sum = 0.
      train_IoU_sum = 0.
 
    # update global steps.
    global_step.assign_add(1)

    # Save checkpoints.
    if (step + 1) % SAVE_STEP == 0:
      print('evaluating...')
      eval_list =random.sample(file_list[:len(file_list)//RATIO_EVAL], 256)
      eval_bpp_ae, eval_bpp_hyper, eval_IoU = eval(eval_list, batch_size=8)
      print("Bpps: {0:.4f} + {1:.4f}".format(eval_bpp_ae, eval_bpp_hyper))
      print("IoU: {0:.4f}".format(eval_IoU))

      with eval_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('bpp_ae', eval_bpp_ae)
        tf.contrib.summary.scalar('bpp_hyper', eval_bpp_hyper)
        tf.contrib.summary.scalar('bpp', eval_bpp_ae + eval_bpp_hyper)
        tf.contrib.summary.scalar('IoU', eval_IoU)

      checkpoint.save(file_prefix = checkpoint_prefix)

# checkpoint.
checkpoint_dir = os.path.join('./checkpoints', 
  args.prefix+'hyper|a{0:.2f}b{1:.2f}/'.format(alpha, beta))
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
  status = checkpoint.restore(ckpt.model_checkpoint_path)
  print('Loading checkpoints...')
elif init_ckpt_dir !='':
  init_ckpt = tf.train.latest_checkpoint(checkpoint_dir=init_ckpt_dir)
  print('init_ckpt: ', init_ckpt)
  status = checkpoint.restore(init_ckpt)
  global_step.assign(0)
  print('Loading initial checkpoints from {}...'.format(init_ckpt_dir))


log_dir = os.path.join('./logs', \
  args.prefix+'hyper_a{0:.2f}b{1:.2f}/'.format( \
  alpha, beta))

eval_log_dir = os.path.join('./logs', \
  args.prefix+'hyper_eval_a{0:.2f}b{1:.2f}/'.format( \
  alpha, beta))

writer = tf.contrib.summary.create_file_writer(log_dir)
eval_writer = tf.contrib.summary.create_file_writer(eval_log_dir)

if __name__ == "__main__":
  train()
  
