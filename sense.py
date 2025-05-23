import sigpy as sp
import sigpy.mri as mr
import numpy as np
import h5py
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from utils import psnr, ssim
from deepdwi.models import mri
# Assume you already have these:
# kspace: ndarray of shape (num_coils, height, width), complex64 or complex128
# csm: coil sensitivity maps (same shape as kspace)
# mask: binary mask (height, width) → optional

def normalize_np(data):
    vmax = np.abs(data).max()
    data /= (vmax + 1e-5)
    return data

def center_crop(tensor, target_height=200, target_width=200):
    h, w = tensor.shape
    top = (h - target_height) // 2
    left = (w - target_width) // 2
    return tensor[top:top + target_height, left:left + target_width]

def undersample_(c_gt, csm, mask, sigma):

    ncoil, nrow, ncol = csm.shape
    csm = csm[None, ...]  # 4dim
    mask = mask[None, ...] # 3dim
    c_gt = c_gt[None, ...] # 3dim

    SenseOp = mri.SenseOp(csm, mask)

    b = SenseOp.fwd(c_gt)

    noise = torch.randn(b.shape) + 1j * torch.randn(b.shape)
    noise = noise * sigma / (2.**0.5)

    atb = SenseOp.adj(b + noise).squeeze(0).detach().numpy()

    return atb
    
poisson_mask = './data/mask_poisson_accelx16_396_768.mat'
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import r2c, c2r, fft_torch

recon_dm = np.load('recon_dm_049_v3.npy')
print("recon_dm:", recon_dm.shape)

file = loadmat(poisson_mask)
mask = file['mask']
 
root = './data/fastmri_dataset.hdf5'
h5_file = h5py.File(root, 'r')
# print(h5_file.keys()) 'trnCsm', 'trnKspace', 'trnOrg', 'tstCsm', 'tstKspace', 'tstOrg'
trnOrg = h5_file['trnOrg'] # (2150, 396, 768)
tstOrg = h5_file['tstOrg'] # (542, 396, 768)

trnKspace = h5_file['trnKspace'] # (2150, 16, 396, 768)
trnCsm = h5_file['trnCsm'] # (2150, 16, 396, 768)

DIR = os.path.dirname(os.path.realpath(__file__))
HOME_DIR = DIR #DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

HOME_DIR = Path(HOME_DIR)
(HOME_DIR / 'figures/fastmri_sense').mkdir(parents=True, exist_ok=True)

index = 49
y = trnOrg[index]

recon = mr.app.SenseRecon(
    y=trnKspace[index] * mask,
    mps=trnCsm[index],
    lamda=1e-3,           # regularization weight
    max_iter=30,          # number of iterations
    show_pbar=True
)

# Run reconstruction
y_pred = recon.run()

# MoDL
device = 'cuda'
from models.modl import MoDL
n_layers, k_iters = 1, 1
model = MoDL(n_layers, k_iters)
model.to(device)
checkpoints_dir = '/home/woody/iwbi/iwbi102h/fastMRI_recon_models/workspace/fastmri_modl,k=1,n=1/checkpoints/best.epoch0013-score36.6389.pth'
state_dict = torch.load(checkpoints_dir)
model.load_state_dict(state_dict)

gt = trnOrg[index]
csm = trnCsm[index]
x = undersample_(gt, csm, mask, 0.01)
print(x.dtype, csm.dtype, mask.dtype)
x, csm, mask = torch.from_numpy(c2r(x)).to(device), torch.from_numpy(csm).to(device), torch.from_numpy(mask).to(device)
y_modl, y_dn = model(x[None, ...], csm[None, ...], mask[None, ...])

y_modl = y_modl[0].cpu().detach().numpy()


y = y[6: 390, 192: 576]
y_pred = y_pred[6: 390, 192: 576]
y_modl = y_modl[6: 390, 192: 576]

y = normalize_np(center_crop(y, target_height=200, target_width=200))
y_pred = normalize_np(center_crop(y_pred, target_height=200, target_width=200))
y_dm = normalize_np(center_crop(recon_dm, target_height=200, target_width=200))
y_modl = normalize_np(center_crop(y_modl, target_height=200, target_width=200))

plt.imsave(HOME_DIR / 'figures/fastmri_sense' / 'recon.png', np.abs(y_pred), cmap='gray', vmin=np.abs(y_pred).min(), vmax=np.abs(y_pred).max())
plt.imsave(HOME_DIR / 'figures/fastmri_sense' / 'org.png', np.abs(y), cmap='gray', vmin=np.abs(y).min(), vmax=np.abs(y).max())
plt.imsave(HOME_DIR / 'figures/fastmri_sense' / 'recon_dm.png', np.abs(y_dm), cmap='gray', vmin=np.abs(y_dm).min(), vmax=np.abs(y_dm).max())
plt.imsave(HOME_DIR / 'figures/fastmri_sense' / 'recon_modl.png', np.abs(y_modl), cmap='gray', vmin=np.abs(y_modl).min(), vmax=np.abs(y_modl).max())

psnr_sense = psnr(np.abs(y), np.abs(y_pred), data_range=1.0)
ssim_sense = ssim(np.abs(y), np.abs(y_pred), data_range=1.0)
print("psnr:", psnr_sense)
print("ssim:", ssim_sense)

psnr_sense = psnr(np.abs(y), np.abs(y_dm), data_range=1.0)
ssim_sense = ssim(np.abs(y), np.abs(y_dm), data_range=1.0)
print("psnr:", psnr_sense)
print("ssim:", ssim_sense)

psnr_sense = psnr(np.abs(y), np.abs(y_modl), data_range=1.0)
ssim_sense = ssim(np.abs(y), np.abs(y_modl), data_range=1.0)
print("psnr:", psnr_sense)
print("ssim:", ssim_sense)