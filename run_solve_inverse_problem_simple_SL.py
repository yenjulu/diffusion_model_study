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

import controllable_generation_TV_simple
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
def add_noise(batch, sde, ratio): # batch: (s, 2 * d, h, w) 
    eps = 1e-5
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z * ratio
    return perturbed_data


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


def prep_model_inputs_default(setting, kdat, coil):  
    from deepdwi import prep  
    coil4, kdat6, kdat_scaling, phase_shot, phase_slice, mask, DWI_MUSE = \
        prep.prep_dwi_data(data_file=kdat, coil_file=coil,
                        slice_idx=setting["slice_idx"],
                        norm_kdat=1.0,
                        N_shot_retro=setting["N_shot_retro"], # 0, 1, 2, 4
                        N_diff_split=setting["N_diff_split"],
                        N_diff_split_index=setting["N_diff_split_index"],
                        redu=setting["redu"],                          
                        return_muse=True)

    train_mask = lossf_mask = mask7 = mask[np.newaxis]

    coil7, kdat7, phase_shot7, phase_slice7 = \
        repeat_data(coil4, kdat6, phase_shot, phase_slice,
                    N_repeats=1)

    print(f">>> coil7 shape\t: {coil7.shape}, type: {coil7.dtype}")
    print(f">>> kdat7 shape\t: {kdat7.shape}, type: {kdat7.dtype}")
    print(f">>> phase_shot7 shape\t: {phase_shot7.shape}, type: {phase_shot7.dtype}")
    print(f">>> phase_slice7 shape\t: {phase_slice7.shape}, ' type: {phase_slice7.dtype}")
    print(f">>> mask7 shape\t: {mask7.shape}, type: {mask7.dtype}")
 
    print(f"Input data for model generated succesfully")
    
    coil7, kdat7, phase_shot7, phase_slice7 = [x.cpu().numpy() for x in [coil7, kdat7, phase_shot7, phase_slice7]]

    inputs = {
        'sens_real': np.real(coil7).astype(np.float32),
        'sens_imag': np.imag(coil7).astype(np.float32),
        'kspace_real': np.real(kdat7).astype(np.float32),
        'kspace_imag': np.imag(kdat7).astype(np.float32),
        'train_mask': train_mask.astype(np.float32),  
        'lossf_mask': lossf_mask.astype(np.float32),
        'phase_echo_real': np.real(phase_shot7).astype(np.float32),
        'phase_echo_imag': np.imag(phase_shot7).astype(np.float32),
        'phase_slice_real': np.real(phase_slice7).astype(np.float32),
        'phase_slice_imag': np.imag(phase_slice7).astype(np.float32),
    }

    return inputs, coil7, kdat7, phase_shot7, phase_slice7, train_mask, lossf_mask


# %%
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
num_scales = 6000 
N = num_scales
img_size = 200
num_diffusions = 1

# path = '/home/woody/iwbi/iwbi102h/deepdwi_study/figures/output_images_slic_c_no2'
# vol = '/3_91.npy'
# img = np.load(path + vol)[None, None, ...]
# img = torch.from_numpy(img).to(device)
 
# coil = '/data/1.0mm_126-dir_R3x3_coils.h5'
# kdat = '/data/1.0mm_126-dir_R3x3_kdat_slice_037.h5'   
# vol = 'kdat_slice_037'
coil = '/data/meas_MID00081_FID00082_Seg4_20_1p0iso/coils_reord.h5'
kdat = '/data/meas_MID00081_FID00082_Seg4_20_1p0iso/kdat_slice_032.h5'
vol = 'kdat_slice_032'

'''
setting = {
    "N_diff": 126,
    "batches": 1,
    "recon_kdat_ky": 200,
    "recon_kdat_kx": 200,
    "N_shot": 2,
    "N_shot_retro": 0,
    "N_slices": 114,
    "MB": 3,
    "N_Accel_PE": 2,
    "N_diff_split": 126,
    "N_diff_split_index": 12,
    "slice_idx": 37,
    "redu": 1.0
    }  
'''

setting = {
    "N_diff": 21,
    "batches": 1,
    "recon_kdat_ky": 200,
    "recon_kdat_kx": 200,
    "N_shot": 4,
    "N_slices": 114,
    "N_shot_retro": 2,
    "MB": 3,
    "N_Accel_PE": 1,
    "N_diff_split": 21,
    "N_diff_split_index": 1,
    "slice_idx": 32,
    "redu": 1.0        
    }  


_, coil7, kdat7, phase_shot7, phase_slice7, train_mask, lossf_mask = prep_model_inputs_default(setting, kdat, coil)

coil7 = torch.from_numpy(coil7).to(device)
kdat7 = torch.from_numpy(kdat7).to(device)
phase_shot7 = torch.from_numpy(phase_shot7).to(device)
phase_slice7 = torch.from_numpy(phase_slice7).to(device) 

Sense = mri.Sense(coil7, kdat7,
            phase_echo=phase_shot7, combine_echo=True,
            phase_slice=phase_slice7)   
ATy = Sense.adjoint(Sense.y).squeeze()
img = ATy[None, ...].to(device)
print("img:", img.shape)

# %%
HOME_DIR = DIR = os.path.dirname(os.path.realpath(__file__))

if sde.lower() == 'vesde':
    configs = importlib.import_module(f"configs.ve.{config_name}")

config = configs.get_config()   
    
if config_name == 'dwi_200_ncsnpp_continuous':
    ckpt_filename = f"/home/woody/iwbi/iwbi102h/diffusion_model_study/workdir/20_dir/checkpoints-meta/checkpoint.pth"
print(ckpt_filename)

collapse_slices = 3 #config.training.collapse_slices

config.model.num_scales = num_scales
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sde.N = N
sampling_eps = 1e-5

predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.36
n_steps = 1

# parameters for Fourier CS recon
mask_type = 'gaussian2d'  # gaussian1d, uniformrandom2d, gaussian2d, uniform1d
use_measurement_noise = False
acc_factor = 8.0
center_fraction = 0.25

# ADMM TV parameters lamb < rho
rho_list = [0.01]
lamb_list = [0.001] 
# lamb_list: 0.001 with rho_list: 0.01 
 
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
b = img.shape[0]

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
        
        img_clear = img
        
        # add noise
        # img = kspace_to_nchw(torch.view_as_real(img).permute(1, 0, 2, 3, 4).contiguous()) # (s, 2 * d, h, w)
        # img = add_noise(img, sde, 6e-3) # batch: (s, 2 * d, h, w) : 0.6, 0.4
        # img = torch.view_as_complex(nchw_to_kspace(img).permute(1, 0, 2, 3, 4).contiguous()) # (d, s, h, w)
                   
        # forward model
        kspace = fft2(img) # (d, s, h, w)
        d, s, h, w = kspace.shape

        # generate mask
        mask_default = get_mask(torch.zeros(num_diffusions, 1, h, w), img_size, num_diffusions,
                        type=mask_type, acc_factor=acc_factor, center_fraction=center_fraction)
        mask_default = mask_default.to(img.device)
        print("mask_default:", mask_default.shape)  

        sizes = (collapse_slices, num_diffusions, img_size)  
                        
        pc_fouriercs = controllable_generation_TV_simple.get_pc_radon_ADMM_TV_mri(Sense, score_model, device, sde, predictor, corrector, inverse_scaler,                                                                       
                                                                        snr=snr, lamb_1=lamb, rho=rho, n_steps=n_steps, probability_flow=probability_flow, save_progress=False, continuous=config.training.continuous,
                                                                        mask_default=mask_default, sizes=sizes, img=img)
                                                                                
        recon_img = pc_fouriercs()
  
        print("recon_img:", recon_img.shape) 
        count = 0

        for d in range(recon_img.shape[0]):           
            for s in range(recon_img.shape[1]):    
                recon_img_ds = normalize(recon_img[d][s])
                aty_img_ds = normalize(img[d][s])
                plt.imsave(save_root / 'label' / f'{count}.png', clear(torch.abs(aty_img_ds), normalize=False), cmap='gray', vmin=0, vmax=torch.amax(torch.abs(aty_img_ds)) * 0.6)
                plt.imsave(save_root / 'recon' / f'{count}.png', clear(torch.abs(recon_img_ds), normalize=False), cmap='gray', vmin=0, vmax=torch.amax(torch.abs(recon_img_ds)) * 0.6)
                count += 1