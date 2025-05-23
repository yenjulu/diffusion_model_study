import numpy as np
import h5py
import math
import h5py
import torch
import os
from utils import clear
from pathlib import Path
import matplotlib.pyplot as plt

def crop(x, crop_size=96): # 192, 96
    start = (x.shape[-2] - crop_size) // 2 
    end = (x.shape[-1] - crop_size) // 2    
    return x[..., start:start + crop_size, end:end + crop_size]

def ssim(y, y_pred):
    from skimage.metrics import structural_similarity
    return structural_similarity(y, y_pred, data_range=y.max() - y.min())
    
def mse(y, y_pred):
    # Assuming y and y_pred are numpy arrays with complex values
    # Calculate MSE as the mean of the squared absolute differences
    return np.mean(np.abs(y - y_pred)**2)

def psnr(y, y_pred):
    # Calculate MSE using the modified function
    error = mse(y, y_pred)
    # Use numpy's sqrt which supports complex numbers (if needed)
    rmse = np.sqrt(error)
    # Maximum possible pixel value of the image (adjust if necessary)
    max_pixel = 1.0
    # Calculate PSNR, typically calculated using real numbers only
    psnr_value = 20 * np.log10(max_pixel / rmse)
    return psnr_value.real  # Return the real part if all you need is the PSNR

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

def normalize(data):
    vmax = np.abs(data).max()
    data /= (vmax + 1e-5)
    return data

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR
print('> HOME: ', HOME_DIR)
HOME_DIR = Path(HOME_DIR)
'''

'''
HOME_DIR = '/home/woody/iwbi/iwbi102h/DiffusionModel'
FILE_PATH = '/results/dwi_200_ncsnpp_continuous/Fourier_DWI_admm_tv/gaussian2d/acc2.0/lamb0.0005/rho0.008/zsssl_slice_037.h5'

label = np.load(HOME_DIR + FILE_PATH + '/label/4.npy')
recon = np.load(HOME_DIR + FILE_PATH + '/recon/4.npy')
# print(label.shape, recon.shape)
recon = normalize(crop(recon, crop_size=192))
label = normalize(crop(label, crop_size=192))

snr_recon = psnr(label, recon)
print('snr_recon', snr_recon)
print("--------------------------------")
ssim_recon = ssim(np.abs(label), np.abs(recon))
print('ssim_recon', ssim_recon)

