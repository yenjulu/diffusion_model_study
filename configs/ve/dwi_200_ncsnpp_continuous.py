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

# Lint as: python3
"""Training NCSN++ on fastmri knee with VE SDE."""

from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  
  method = config.method
  method.shepp_logan = False  # True: model trained with sheep logan data; False: model trained with DWI data
  
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.collapse_slices = 1  #   training.collapse_slices = 3
  
  if not method.shepp_logan:
    training.batch_size = 1 # = number of diffusion directions 
  else:
    training.batch_size = 1 


  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data
  data = config.data
  data.dataset = 'fastmri_knee'
  
  if not method.shepp_logan:
    # data.directory = '/home/woody/iwbi/iwbi102h/deepdwi_study/examples/2025-03-21_zsssl_1.0mm_21-dir_R1x3_kdat_slice_009_shot-retro-2_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss'
    data.directory = '/home/woody/iwbi/iwbi102h/deepdwi_study/'
    data.root = data.directory + '/for_dm_trn_images.h5'  # meas_MID00081_z2_images
  else:
    data.directory = '/home/woody/iwbi/iwbi102h/deepdwi_study/data'
    data.root = data.directory + '/shepp_logan_192.h5'  
  
  data.is_complex = True
  data.is_multi = False
  data.magpha = False
  
  if not method.shepp_logan:
    data.image_size = 384
    data.num_channels = 2 * training.batch_size
  else:
    data.image_size = 192
    data.num_patches = 16 # 4 x 4 = 16; 2 x 2 = 4 
    data.patch_size = 48 # 48, 24, 12, 6; 
    data.num_channels = 2 * data.num_patches # 2 * 16   

  # model
  model = config.model
  model.name = 'ncsnpp'  # ncsnpp
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (25,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config
