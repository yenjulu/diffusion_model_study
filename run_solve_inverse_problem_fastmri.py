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

import controllable_generation_TV_fastmri
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

from fastmri import mri
from deepdwi import util, prep
from deepdwi.recons import zsssl

# %%
# %%
def normalize_np(data):
    vmax = np.abs(data).max()
    data /= (vmax + 1e-5)
    return data
        
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
num_scales = 3000 
N = num_scales
img_size = 384
num_diffusions = 1

poisson_mask = '/home/woody/iwbi/iwbi102h/diffusion_model_fastmri_study/data/mask_poisson_accelx16_396_768.mat'
file = loadmat(poisson_mask)
mask = file['mask']
 
root = '/home/woody/iwbi/iwbi102h/diffusion_model_fastmri_study/data/fastmri_dataset.hdf5'
h5_file = h5py.File(root, 'r')

trnOrg = h5_file['trnOrg']
trnKspace = h5_file['trnKspace'] 
trnCsm = h5_file['trnCsm']

vol = 'fastmri_diffusion_model'
index = 49

coil = torch.from_numpy(trnCsm[index]).unsqueeze(0).to(device)
kdat = torch.from_numpy(trnKspace[index]).unsqueeze(0).to(device)
mask = torch.from_numpy(mask).unsqueeze(0).to(device)

print("coil:", coil.shape)
print("kdat:", kdat.shape)
print("mask:", mask.shape)



# %%
HOME_DIR = DIR = os.path.dirname(os.path.realpath(__file__))

if sde.lower() == 'vesde':
    configs = importlib.import_module(f"configs.ve.{config_name}")

config = configs.get_config()   
    
if config_name == 'dwi_200_ncsnpp_continuous':
    ckpt_filename = f"/home/woody/iwbi/iwbi102h/diffusion_model_fastmri_study/workdir/fastmri/checkpoints-meta/checkpoint.pth"
    # ckpt_filename = f"/home/woody/iwbi/iwbi102h/diffusion_model_study/workdir/20_dir/checkpoints-meta/checkpoint.pth"
print(ckpt_filename)

collapse_slices = 1

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
# rho_list = [0.01]
# lamb_list = [0.001] 
 
rho_list = [0.008] 
lamb_list = [0.0001]
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

        # generate mask
        mask_default = get_mask(torch.zeros(num_diffusions, 1, img_size, img_size), img_size, num_diffusions,
                        type=mask_type, acc_factor=acc_factor, center_fraction=center_fraction)
        mask_default = mask_default.to(config.device)
        print("mask_default:", mask_default.shape)  

        sizes = (collapse_slices, num_diffusions, img_size)  
        
        Sense = mri.SenseOp(coil, mask)
        img = Sense.adj(kdat * mask).unsqueeze(0) 
        img = img[:, :, 6: 390, 192: 576] 
        img = normalize(img)
                      
        pc_fouriercs = controllable_generation_TV_fastmri.get_pc_radon_ADMM_TV_mri(Sense, score_model, device, sde, predictor, corrector, inverse_scaler,                                                                       
                                                                        snr=snr, lamb_1=lamb, rho=rho, n_steps=n_steps, probability_flow=probability_flow, save_progress=False, continuous=config.training.continuous,
                                                                        mask_default=mask_default, sizes=sizes, img=img)
                                                                                
        recon_img = pc_fouriercs()
  
        print("recon_img:", recon_img.shape) 
   
        recon_img_ds = normalize(recon_img[0][0])
        aty_img_ds = normalize(img[0][0])
        plt.imsave(save_root / 'label' / 'aty.png', clear(torch.abs(aty_img_ds), normalize=False), cmap='gray', vmin=torch.amin(torch.abs(aty_img_ds)), vmax=torch.amax(torch.abs(aty_img_ds)))
        plt.imsave(save_root / 'recon' / 'recon_dm.png', clear(torch.abs(recon_img_ds), normalize=False), cmap='gray', vmin=torch.amin(torch.abs(recon_img_ds)), vmax=torch.amax(torch.abs(recon_img_ds)))
        np.save(str(save_root / 'recon' / 'recon_dm.npy'), clear(recon_img_ds, normalize=False))