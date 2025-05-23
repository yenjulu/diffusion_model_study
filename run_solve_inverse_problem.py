import sys
import os

# inverse problem solver
from pathlib import Path
from itertools import islice
import datasets
import time
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import functools
import h5py
from scipy.io import savemat, loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib
from torch.utils.data import DataLoader

import controllable_generation_TV
from ml_collections import ConfigDict
from utils import restore_checkpoint, fft2, ifft2, show_samples_gray, get_mask, clear, kspace_to_nchw, nchw_to_kspace
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
from models import ncsnpp
from models import utils as mutils

from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                    LangevinCorrector,
                    LangevinCorrectorCS)

from deepdwi.models import mri
from deepdwi import util, prep
from deepdwi.recons import zsssl

# %%
def prep_mask(mask: np.ndarray, N_repeats: int = 12,
              valid_rho: float = 0.2,
              train_rho: float = 0.4):
    mask = torch.from_numpy(mask)
    res_mask, valid_mask = zsssl.uniform_samp(mask, rho=valid_rho) # train_mask, lossf_mask
    valid_mask = valid_mask[None, ...]  # 7dim

    train_mask = []
    lossf_mask = []

    for r in range(N_repeats):

        train_mask1, lossf_mask1 = zsssl.uniform_samp(res_mask, rho=train_rho)

        train_mask.append(train_mask1)
        lossf_mask.append(lossf_mask1)

    train_mask = torch.stack(train_mask)
    lossf_mask = torch.stack(lossf_mask)

    f = h5py.File(DIR + '/mask.h5', 'w')
    f.create_dataset('train', data=train_mask.detach().cpu().numpy())
    f.create_dataset('lossf', data=lossf_mask.detach().cpu().numpy())
    f.create_dataset('valid', data=valid_mask.detach().cpu().numpy())
    f.close()

    return mask, train_mask, lossf_mask, valid_mask

def add_noise(batch, sde, ratio): # batch: (s, 2 * d, h, w) 
    eps = 1e-5
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z * ratio
    return perturbed_data

# %%
def repeat_data(coil4: np.ndarray,
                kdat6: np.ndarray,
                phase_shot: np.ndarray,
                phase_slice: np.ndarray,
                N_repeats: int = 12):

    coil7 = torch.from_numpy(coil4)
    coil7 = coil7[None, None, None, ...]
    coil7 = torch.tile(coil7, tuple([N_repeats] + [1] * (coil7.dim()-1)))

    kdat7 = torch.from_numpy(kdat6)
    kdat7 = kdat7[None, ...]
    kdat7 = torch.tile(kdat7, tuple([N_repeats] + [1] * (kdat7.dim()-1)))

    if phase_shot is not None:
        phase_shot7 = torch.from_numpy(phase_shot)
        phase_shot7 = phase_shot7[None, ...]
        phase_shot7 = torch.tile(phase_shot7, tuple([N_repeats] + [1] * (phase_shot7.dim()-1)))
    else:
        phase_shot7 = None

    if phase_slice is not None:
        phase_slice7 = torch.from_numpy(phase_slice)
        phase_slice7 = phase_slice7[None, None, None, None, ...]
        phase_slice7 = torch.tile(phase_slice7, tuple([N_repeats] + [1] * (phase_slice7.dim()-1)))
    else:
        phase_slice7 = None

    return coil7, kdat7, phase_shot7, phase_slice7

def normalize_np(data):
    vmax = np.abs(data).max()
    data /= (vmax + 1e-5)
    return data

def plot_images(diff_idx, image_list, name_list, saved_name, layout, save_dir):
        import math
        num_images = len(image_list)
        rows = layout[0]
        cols = layout[1]
        plt.figure(figsize=(6, 4)) 
        
        for i, (image, name) in enumerate(zip(image_list, name_list)):               
            plt.subplot(rows, cols, i + 1)
            image = normalize_np(image)
            # image = crop(image, crop_size=200)  ## cropping 
            magnitude = np.abs(image)
            magnitude = np.flipud(magnitude)
            
            threshold = np.percentile(magnitude, 10)
            magnitude_thresholded = np.where(magnitude < threshold, 0, magnitude)
            # vmax=np.amax(magnitude_thresholded) * 0.6
            vmax=np.amax(magnitude) * 0.6
            
            plt.imshow(magnitude, cmap='gray', vmin=0, vmax=vmax, interpolation='None')
            plt.title(f"{name}", fontsize = 7)             
            plt.axis('off') 
        
        plt.figtext(0.5, 0.01, f"diffusion_index: {diff_idx}", ha="center", fontsize=10)    
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.01, hspace=0.2)
        plt.savefig(f'{save_dir}/{saved_name}.png', dpi=800)
        plt.close()
        
def normalize(data):
    vmax = torch.abs(data).max()
    data /= (vmax + 1e-5)
    return data
###############################################
# Configurations
###############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

problem = 'Fourier_DWI_admm_tv'
config_name = 'dwi_200_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2003 # 
 
N = num_scales
img_size = 200
shepp_logan_img_size = 192
num_diffusions = 1
Sense_recon = False  # True, False: use the default one
shepp_logan = False
shepp_logan_input = False

# root = '/home/woody/iwbi/iwbi102h/deepdwi_study/examples/zsssl_1.0mm_126-dir_R3x3_kdat_slice_037_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss' 
root = '/home/woody/iwbi/iwbi102h/deepdwi_study/examples/2025-03-21_zsssl_1.0mm_21-dir_R1x3_kdat_slice_009_shot-retro-2_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss'   
# vol = '/zsssl_slice_037.h5'
vol = '/zsssl_slice_009_diff_split_006.h5'   
ckt_path = root + '/zsssl_best.pth'

# shepp logan input data
if shepp_logan_input:
    root = '/home/woody/iwbi/iwbi102h/deepdwi_study/data'
    vol = '/shepp_logan.h5' # phantom_img.h5  shepp_logan.h5


# %%
HOME_DIR = DIR = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='run zsssl.')

parser.add_argument('--config', type=str,
                    default='/home/woody/iwbi/iwbi102h/deepdwi_study/configs/zsssl_redu_1p0_2.yaml',
                    help='yaml config file for zsssl.')

parser.add_argument('--N_shot_retro', type=int,
                    default=0,
                    help='retro. undersample the number of shots')

args = parser.parse_args()
   
with open(args.config, 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

data_conf = config_dict.get('data', {})
model_conf = config_dict.get('model', {})

coil4, kdat6, kdat_scaling, phase_shot, phase_slice, mask, DWI_MUSE = \
    prep.prep_dwi_data(data_file=data_conf['kdat'],
                        navi_file=data_conf['navi'],
                        coil_file=data_conf['coil'],
                        slice_idx=data_conf['slice_idx'],
                        norm_kdat=data_conf['normalize_kdat'],
                        N_shot_retro=args.N_shot_retro,
                        N_diff_split=data_conf['N_diff_split'],
                        N_diff_split_index=data_conf['N_diff_split_index'],
                        redu=data_conf['redu'],
                        return_muse=True)

mask, train_mask, lossf_mask, valid_mask = \
    prep_mask(mask, N_repeats=data_conf['repeats'],
                valid_rho=data_conf['valid_rho'],
                train_rho=data_conf['train_rho'])

coil7, kdat7, phase_shot7, phase_slice7 = \
    repeat_data(coil4, kdat6, phase_shot, phase_slice,
                N_repeats=data_conf['repeats'])

print('>>> train_mask[[0]] shape\t: ', train_mask[[0]].shape, ' type: ', train_mask.dtype)
print('>>> mask[[0]] shape\t: ', mask[[0]].shape, ' type: ', mask.dtype)
print('>>> coil7[[0]] shape\t: ', coil7[[0]].shape, ' type: ', coil7.dtype)
print('>>> kdat7[[0]] shape\t: ', kdat7[[0]].shape, ' type: ', kdat7.dtype)
print('>>> phase_shot7[[0]] shape\t: ', phase_shot7[[0]].shape, ' type: ', phase_shot7.dtype)
print('>>> phase_slice7[[0]] shape\t: ', phase_slice7[[0]].shape, ' type: ', phase_slice7.dtype)

'''
train_mask[[0]] shape    :  torch.Size([1, 14, 2, 1, 1, 200, 200])  type:  torch.float64
mask[[0]] shape     :  torch.Size([1, 2, 1, 1, 200, 200])  type:  torch.float64
coil7[[0]] shape    :  torch.Size([1, 1, 1, 32, 3, 200, 200])  type:  torch.complex64
kdat7[[0]] shape    :  torch.Size([1, 2, 2, 32, 1, 200, 200])  type:  torch.complex64
phase_shot7[[0]] shape      :  torch.Size([1, 2, 2, 1, 3, 200, 200])  type:  torch.complex128
phase_slice7[[0]] shape     :  torch.Size([1, 1, 1, 1, 3, 200, 200])  type:  torch.complex128
all_img: torch.Size([2, 3, 200, 200])
'''

if sde.lower() == 'vesde':
    configs = importlib.import_module(f"configs.ve.{config_name}")

config = configs.get_config()   
    
if config_name == 'dwi_200_ncsnpp_continuous' and not shepp_logan:
    ckpt_filename = f"/home/woody/iwbi/iwbi102h/diffusion_model_study/workdir/00081/checkpoints-meta/checkpoint.pth"
    # ckpt_filename = f"/home/woody/iwbi/iwbi102h/diffusion_model_study/workdir/DWI_9d1s/checkpoints-meta/checkpoint.pth"
if config_name == 'dwi_200_ncsnpp_continuous' and shepp_logan:
    # ckpt_filename = f"/home/woody/iwbi/iwbi102h/diffusion_model_study/workdir/shepp_logan_patches_4/checkpoints-meta/checkpoint.pth"
    ckpt_filename = f"/home/woody/iwbi/iwbi102h/diffusion_model_study/workdir/shepp_logan_patches_16/checkpoints-meta/checkpoint.pth"
    # ckpt_filename = f"/home/woody/iwbi/iwbi102h/diffusion_model_study/workdir/shepp_logan_patches_64/checkpoints-meta/checkpoint.pth"

print(ckpt_filename)

if shepp_logan:
    patch_size = config.data.patch_size

collapse_slices = config.training.collapse_slices

config.model.num_scales = num_scales
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sde.N = N
sampling_eps = 1e-5

predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

# parameters for Fourier CS recon
mask_type = 'gaussian2d'  # gaussian1d, uniformrandom2d, gaussian2d, uniform1d
use_measurement_noise = False
acc_factor = 2.0
center_fraction = 0.15

# ADMM TV parameters lamb < rho
# lamb_list = [0.005] 
# rho_list = [0.01]
lamb_list = [0.0001] 
rho_list = [0.008] 
    
if Sense_recon:
    lamb_list = [0.0001] 
    rho_list = [0.008] 

# if Sense_recon:
#     lamb_list = [0.005] 
#     rho_list = [0.008] 
    
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                            decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
            model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
ema.copy_to(score_model.parameters())

h5_file = h5py.File(root + vol, 'r')
all_img = np.squeeze(h5_file['ZS'][:])

if shepp_logan_input:
    _, h, w = all_img.shape
    all_img = all_img[:, None, ...]
    all_img = all_img.reshape((-1, 3, h, w))
    print('all_img:', all_img.shape)

N_diff = 3
N_diff_split = 3 # data_conf['N_diff_split']
N_diff_split_index = 2 # data_conf['N_diff_split_index']

if N_diff_split > 1:
    N_diff_sub = N_diff // N_diff_split
    diff_idx = range(N_diff_split_index * N_diff_sub,
                        (N_diff_split_index+1) * N_diff_sub if N_diff_split_index < N_diff_split else N_diff)

    all_img = all_img[diff_idx, ...]
    assert len(diff_idx) == num_diffusions

all_img = torch.from_numpy(all_img).squeeze().to(torch.complex64).contiguous()
# all_img = all_img.repeat(3, 1, 1, 1)


if len(all_img.shape) == 3:
    all_img = all_img.unsqueeze(0)
        
d, s, h, w = all_img.shape
print("all_img:", all_img.shape)

for d in range(all_img.shape[0]):
    for s in range(all_img.shape[1]):
        all_img[d][s] = normalize(all_img[d][s])

img = all_img.to(config.device) # (d, s, h, w)

b = img.shape[0]

train_mask_0 = mask[np.newaxis].to(img.device)
coil7_0 = coil7[[0]].to(img.device)
kdat7_0 = kdat7[[0]].to(img.device)
phase_shot7_0 = phase_shot7[[0]].to(img.device)
phase_slice7_0 = phase_slice7[[0]].to(img.device)

for lamb in lamb_list:
    for rho in rho_list:
        print(f'lambda: {lamb}')
        print(f'rho:    {rho}')
        # Specify save directory for saving generated samples
        save_root = Path(f'./results/{config_name}/{problem}/{mask_type}/acc{acc_factor}/lamb{lamb}/rho{rho}/{vol}')
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['input', 'recon', 'label']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)

        ###############################################
        # Inference
        ###############################################
        seed = 20 
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # Sets the seed for current GPU
            torch.cuda.manual_seed_all(seed)  # Sets the seed for all GPUs

        # Ensure deterministic behavior (optional but useful)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # image from Sense ATy 
        # Sense = mri.Sense(coil7_0, kdat7_0,
        #     phase_echo=phase_shot7_0, combine_echo=True,
        #     phase_slice=phase_slice7_0)
        # ATy = Sense.adjoint(Sense.y).squeeze()
        # img = ATy 
        
        img_clear = img
        
        # add noise
        img = kspace_to_nchw(torch.view_as_real(img).permute(1, 0, 2, 3, 4).contiguous()) # (s, 2 * d, h, w)
        for _ in range(10): # int(num_scales)  1500(1e-4): for fake noisy img
            img = add_noise(img, sde, 0.8e-3) # batch: (s, 2 * d, h, w) 
        img = torch.view_as_complex(nchw_to_kspace(img).permute(1, 0, 2, 3, 4).contiguous()) # (d, s, h, w)
                   
        # forward model
        kspace = fft2(img) # (d, s, h, w)
        d, s, h, w = kspace.shape

        # generate mask
        mask_default = get_mask(torch.zeros(num_diffusions, 1, h, w), img_size, num_diffusions,
                        type=mask_type, acc_factor=acc_factor, center_fraction=center_fraction)
        mask_default = mask_default.to(img.device) # (d, 1, h, w)
        print("mask_default:", mask_default.shape)  

        if not shepp_logan:
            sizes = (collapse_slices, num_diffusions, img_size)  
        else:
            sizes = (collapse_slices * num_diffusions, patch_size, shepp_logan_img_size)
        
        ## solve_in_admm=False, Sense_recon=False, img=None                                         
        pc_fouriercs = controllable_generation_TV.get_pc_radon_ADMM_TV_mri(score_model, ckt_path, device, model_conf, train_mask_0, kdat7_0, coil7_0, phase_shot7_0, phase_slice7_0, sde, predictor, corrector, inverse_scaler,                                                                       
                                                                        snr=snr, lamb_1=lamb, rho=rho, n_steps=n_steps, probability_flow=probability_flow, save_progress=False, continuous=config.training.continuous,
                                                                        mask_default=mask_default, shepp_logan=shepp_logan, sizes=sizes,                                                            
                                                                        solve_in_admm=False, Sense_recon=False, img=img)
                                                                                
        x = pc_fouriercs()

        img_clear = img_clear.view(d * s, h, w)

        img_label = img.view(d * s, h, w)
        print("img_label:", img_label.shape)  # (d * s, h, w)
        
        # x = x.squeeze() # (d, s, h, w)
        print("x.shape", x.shape) #  (1, 14, 1, 1, 3, 200, 200)
        
        count = 0
        recon_img = x
 
        for d in range(recon_img.shape[0]):           
            for s in range(recon_img.shape[1]):    
                recon_img_ds = normalize(recon_img[d][s])
                plt.imsave(save_root / 'input' / f'{count}.png', clear(torch.abs(img_clear[count]), normalize=False), cmap='gray', vmin=0, vmax=torch.amax(torch.abs(img_clear[count])) * 0.6) 
                plt.imsave(save_root / 'label' / f'{count}.png', clear(torch.abs(img_label[count]), normalize=False), cmap='gray', vmin=0, vmax=torch.amax(torch.abs(img_label[count])) * 0.6)
                plt.imsave(save_root / 'recon' / f'{count}.png', clear(torch.abs(recon_img_ds), normalize=False), cmap='gray', vmin=0, vmax=torch.amax(torch.abs(recon_img_ds)) * 0.6)
                np.save(str(save_root / 'label' / f'{count}.npy'), clear(img_label[count], normalize=False))
                np.save(str(save_root / 'recon' / f'{count}.npy'), clear(recon_img_ds, normalize=False))
                count += 1