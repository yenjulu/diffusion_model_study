# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from pathlib import Path

import numpy as np
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, unet
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
#import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch import nn
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint, get_mask, kspace_to_nchw, nchw_to_kspace, root_sum_of_squares
from utils import fft2, ifft2, fft2_m, ifft2_m, clear
import torch
import torch.nn as nn
import torch.nn.functional as F

FLAGS = flags.FLAGS

def normalize(data):
    for i in range(data.shape[0]):
      vmax = torch.abs(data[i]).max()
      data[i] /= (vmax + 1e-5)
    return data

class PatchEncoder(nn.Module):
    def __init__(self, patch_size, image_size):
        super(PatchEncoder, self).__init__()
        self.patch_size = patch_size 
        self.image_size = image_size 
        '''
        input: (B, C, H, W, 2)
        output: (B, C * num_patches, patch_size, patch_size, 2) 
        '''

    def forward(self, x): # x is a complex with shape = (B, C, H, W, 2)
        B, C, H, W, _ = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "H and W must be divisible by patch size"
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Reshape keeping the real and imaginary parts
        x = x.reshape(B, C, num_patches_h, self.patch_size, num_patches_w, self.patch_size, 2)

        # Permute and reshape to merge the patch dimensions into the channel dimension
        x = x.permute(0, 2, 4, 1, 3, 5, 6).reshape(B, C * num_patches, self.patch_size, self.patch_size, 2)       
        return x

    def backward(self, x):  # x is complex with shape = (B, C * num_patches, patch_size, patch_size, 2)
        B, C_patches, patch_h, patch_w, _ = x.shape
        assert patch_h == self.patch_size and patch_w == self.patch_size, "Patch dimensions must match the original patch size"
        
        # Calculate the number of patches and original dimensions
        
        H = self.image_size
        W = self.image_size
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        C = C_patches // num_patches
        
        # Reshape to split patch dimensions into original spatial layout
        x = x.reshape(B, num_patches_h, num_patches_w, C, self.patch_size, self.patch_size, 2)
        
        # Permute to get back to original spatial dimensions
        x = x.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, C, H, W, 2)
        return x

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  Path(sample_dir).mkdir(parents=True, exist_ok=True)

  tb_dir = os.path.join(workdir, "tensorboard")
  Path(tb_dir).mkdir(parents=True, exist_ok=True)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
  Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
  Path(checkpoint_meta_dir).mkdir(parents=True, exist_ok=True)

  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir + '/' + 'checkpoint.pth', state, config.device)
  initial_step = int(state['step'])

  # Build pytorch dataloader for training
  train_dl, eval_dl = datasets.create_dataloader(config)
  num_data = len(train_dl.dataset)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
  if config.method.shepp_logan:
    encoder = PatchEncoder(config.data.patch_size, config.data.image_size)

  # Building sampling functions
  if config.training.snapshot_sampling:
    if not config.method.shepp_logan:
      sampling_shape = (config.training.collapse_slices, config.data.num_channels,
                        config.data.image_size, config.data.image_size)
    else:
      sampling_shape = (config.training.batch_size, config.data.num_channels,
                        config.data.patch_size, config.data.patch_size)

    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  
  print("Batch size:", train_dl.batch_size)
  print("Total number of samples:", len(train_dl.dataset))
  valid_loss_min = np.inf
  
  for epoch in range(1, config.training.epochs):
    print('=================================================')
    print(f'Epoch: {epoch}')
    print('=================================================')
    # # generate mask
    # mask_default = get_mask(torch.zeros(1, 1, 200, 200), 200, 1,
    #                 type='gaussian2d', acc_factor=4.0, center_fraction=0.25)
    # mask_default = mask_default.to(config.device)
    # # print("mask_default:", mask_default.shape)
       
    for step, batch in enumerate(train_dl, start=1):
      if not config.method.shepp_logan:
        # print("batch shape:", batch.shape)
        batch = scaler(batch.to(config.device))  # print("batch shape:", batch.shape) # torch.Size([1, 3, 200, 200])
        # batch = ifft2(fft2(batch) * mask_default)
        batch = kspace_to_nchw(torch.view_as_real(batch).permute(1, 0, 2, 3, 4).contiguous())  # print("batch_nchw shape:", batch.shape) # (3, 2, 200, 200) = (s, 2 * d, h, w)
        # print("batch shape:", batch.shape)
      else:
        batch = scaler(batch.to(config.device))  
        batch = torch.view_as_real(batch)   # (b, c, h, w) --> (b, c, h, w, 2)
        batch = encoder(batch)  # (b, c, h, w, 2) --> (b, c * num_patches, hnew, wnew, 2)
        batch = kspace_to_nchw(batch)  # --> (b, 2 * c * num_patches, hnew, wnew)
        # print("input patches:",  batch.shape)

      
      # Execute one training step
      loss = train_step_fn(state, batch)
      if step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
        global_step = num_data * epoch + step
        writer.add_scalar("training_loss", scalar_value=loss, global_step=global_step)
      if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
        save_checkpoint(checkpoint_meta_dir, state)
      # Report the loss on an evaluation dataset periodically
      if step % config.training.eval_freq == 0:
        eval_batch = scaler(next(iter(eval_dl)).to(config.device))    
        # eval_batch = ifft2(fft2(eval_batch) * mask_default)
        eval_batch = kspace_to_nchw(torch.view_as_real(eval_batch).permute(1, 0, 2, 3, 4).contiguous())
        # print("eval_batch:", eval_batch.shape)
        eval_loss = eval_step_fn(state, eval_batch)
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
        global_step = num_data * epoch + step
        writer.add_scalar("eval_loss", scalar_value=eval_loss.item(), global_step=global_step)
    
    valid_loss = eval_loss.item()
    # Save a checkpoint for every epoch
    if valid_loss <= valid_loss_min:
      valid_loss_min = valid_loss
      save_checkpoint(checkpoint_dir, state, name=f'checkpoint_{epoch}.pth')

    # Generate and save samples for every epoch
    if config.training.snapshot_sampling:
      print('sampling')
      ema.store(score_model.parameters())
      ema.copy_to(score_model.parameters())
      sample, n = sampling_fn(score_model)
      print('sample.shape:', sample.shape) # ([3, 2, 200, 200]) = (s, 2 * d, h, w)
      sample = nchw_to_kspace(sample)  # --> (s, d, h, w, 2)
      if config.method.shepp_logan:
        recon = sample.clone()
        recon = encoder.backward(recon) # (b, num_patches, 48, 48, 2) --> (b, 1, 192, 192, 2)
        recon = root_sum_of_squares(recon, dim=-1) # (b, 1, h, w)
        recon = torch.abs(recon) 
        recon = normalize(recon)
      if config.data.is_complex:
        sample = root_sum_of_squares(sample, dim=-1).unsqueeze(dim=1)
        print('sample.shape:', sample.shape) # [s, 1, d, h, w]
        sample = sample.reshape(shape=(-1, 1, sample.shape[-2], sample.shape[-1]))  # new added
      sample = torch.abs(sample) # get the magnitude
      sample = normalize(sample) # normalize
      ema.restore(score_model.parameters())
      this_sample_dir = os.path.join(sample_dir, "iter_{}".format(epoch))
      Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
      nrow = int(np.sqrt(sample.shape[0]))
      image_grid = make_grid(sample, nrow, padding=2)
      sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
      np.save(os.path.join(this_sample_dir, "sample"), sample)
      save_image(image_grid, os.path.join(this_sample_dir, "sample.png"))
      if config.method.shepp_logan:
        save_image(recon, os.path.join(this_sample_dir, "sample_shepp_logan.png"))
      


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  Path(eval_dir).mkdir(parents=True, exist_ok=True)

  # Build pytorch dataloader for training
  train_dl, eval_dl = datasets.create_dataloader(config)
  num_data = len(train_dl.dataset)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]

      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools

      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())