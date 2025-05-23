import functools
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from numpy.testing._private.utils import measure
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from models import utils as mutils
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
from utils import fft2, ifft2, fft2_m, ifft2_m, clear
from physics.ct import *
from utils import show_samples, show_samples_gray, clear, clear_color, batchfy
from utils import save_checkpoint, restore_checkpoint, get_mask, kspace_to_nchw, nchw_to_kspace, root_sum_of_squares
from deepdwi.models import mri, resnet
from deepdwi import lsqr, util

def normalize(data):
    vmax = torch.abs(data).max()
    data /= (vmax + 1e-5)
    return data

class PatchEncoder(nn.Module):
    def __init__(self, patch_size, image_size):
        super(PatchEncoder, self).__init__()
        self.patch_size = patch_size 
        self.image_size = image_size 
        '''
        input: (B, C, H, W, 2)
        output: (B, C * patches, patch_size, patch_size, 2) 
        '''

    def forward(self, x): # x is a complex with shape = (B, C, H, W, 2)
        B, C, H, W, _ = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "H and W must be divisible by patch size"
        
        patches_h = H // self.patch_size
        patches_w = W // self.patch_size
        patches = patches_h * patches_w
        
        # Reshape keeping the real and imaginary parts
        x = x.reshape(B, C, patches_h, self.patch_size, patches_w, self.patch_size, 2)

        # Permute and reshape to merge the patch dimensions into the channel dimension
        x = x.permute(0, 2, 4, 1, 3, 5, 6).reshape(B, C * patches, self.patch_size, self.patch_size, 2)       
        return x

    def backward(self, x):  # x is complex with shape = (B, C * patches, patch_size, patch_size, 2)
        B, C_patches, patch_h, patch_w, _ = x.shape
        assert patch_h == self.patch_size and patch_w == self.patch_size, "Patch dimensions must match the original patch size"
        
        # Calculate the number of patches and original dimensions
        
        H = self.image_size
        W = self.image_size
        
        patches_h = H // self.patch_size
        patches_w = W // self.patch_size
        patches = patches_h * patches_w
        C = C_patches // patches
        
        # Reshape to split patch dimensions into original spatial layout
        x = x.reshape(B, patches_h, patches_w, C, self.patch_size, self.patch_size, 2)
        
        # Permute to get back to original spatial dimensions
        x = x.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, C, H, W, 2)
        return x

def real_imag_to_complex(real_imag_tensor):
    real_part = real_imag_tensor[..., 0]  
    imag_part = real_imag_tensor[..., 1] 
    complex_tensor = torch.complex(real_part, imag_part)
    return complex_tensor

def crop(x, crop_size=96): # 192, 96
    start = (x.shape[-2] - crop_size) // 2 
    end = (x.shape[-1] - crop_size) // 2    
    return x[..., start:start + crop_size, end:end + crop_size]

def padding(x, pad_size=2): # 4, 2
    # Padding of 2 pixels on each side for 96 -> 100
    return F.pad(x, (pad_size, pad_size, pad_size, pad_size)) 

class lambda_schedule:
  def __init__(self, total=2000):
    self.total = total

  def get_current_lambda(self, i):
    pass

class lambda_schedule_linear(lambda_schedule):
  def __init__(self, start_lamb=1.0, end_lamb=0.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)

class lambda_schedule_const(lambda_schedule):
  def __init__(self, lamb=1.0):
    super().__init__()
    self.lamb = lamb

  def get_current_lambda(self, i):
    return self.lamb

def _Dz(x): # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]
    return y - x

def _DzT(x): # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]

    tempt = -(y-x)
    difft = tempt[:-1]
    y[1:] = difft
    y[0] = x[-1] - x[0]

    return y

def _Dx(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    return y - x

def _DxT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    tempt = -(y - x)
    difft = tempt[:, :, :-1, :]
    y[:, :, 1:, :] = difft
    y[:, :, 0, :] = x[:, :, -1, :] - x[:, :, 0, :]
    return y

def _Dy(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    return y - x

def _DyT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    tempt = -(y - x)
    difft = tempt[:, :, :, :-1]
    y[:, :, :, 1:] = difft
    y[:, :, :, 0] = x[:, :, :, -1] - x[:, :, :, 0]
    return y

def prox_l21(src, lamb, dim):
    """
    src.shape = [448(z), 1, 256(x), 256(y)]
    """
    weight_src = torch.linalg.norm(src, dim=dim, keepdim=True)
    weight_src_shrink = shrink(weight_src, lamb)

    weight = weight_src_shrink / weight_src
    return src * weight

def get_pc_radon_ADMM_TV_mri(model, ckt_path, device, model_conf, mask, kspace, sens, phase_echo, phase_slice, sde, predictor, corrector, inverse_scaler, 
                             n_steps=1, probability_flow=False, continuous=False, snr=None, mask_default=None,
                             denoise=True, eps=1e-5, save_progress=False, save_root=None, sizes=None,
                             lamb_1=5, rho=10, shepp_logan=True, solve_in_admm=False, Sense_recon=False, img=None):
    def _A(x):
        nonlocal mask_default   # mask_default, new_mask .to(dtype=torch.complex64)  
        return fft2(x) * mask_default  

    def _AT(kspace):
        nonlocal mask_default
        return ifft2(kspace) 

    withSense = Sense_recon 
    
    SenseO = mri.Sense(sens, kspace,
                phase_echo=phase_echo, combine_echo=True,
                phase_slice=phase_slice)
            
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    eps = 1e-10
   
    print("mask:", mask.shape)  # torch.Size([1, 14, 2, 1, 1, 200, 200])
    print("kspace:", kspace.shape) # torch.Size([1, 14, 2, 32, 1, 200, 200])

    # if img is None:
    #     kspaceO = kspace
    # else:
    #     d, s, h, w = img.shape
    #     kspaceO = SenseO(img.view(1, d, 1, 1, s, h, w)) 

        
    Sense = mri.Sense(sens, kspace,
                        phase_echo=phase_echo, combine_echo=True, # phase_echo=ones_like, combine_echo=False
                        phase_slice=phase_slice) # , weights=None        

    if img is None:
        measurement = fft2(Sense.adjoint(Sense.y).squeeze())  
    else:
        measurement = fft2(img)
   
    if withSense:
        ATy = Sense.adjoint(Sense.y).squeeze()       
    else:       
        ATy = _AT(measurement * mask_default) 

                      
    ishape = list(Sense.ishape)
    print('>>> ishape to AlgUnroll: ', ishape) # [1, d, 1, 1, s, h, w]

    print("rho & lamda:", rho, lamb_1)
        
    def _Dz(x):  # Batch direction
        y = torch.zeros_like(x)
        y[:-1] = x[1:]
        y[-1] = x[0]
        return y - x

    def _DzT(x):  # Batch direction
        y = torch.zeros_like(x)
        y[:-1] = x[1:]
        y[-1] = x[0]

        tempt = -(y - x)
        difft = tempt[:-1]
        y[1:] = difft
        y[0] = x[-1] - x[0]

        return y

    def shrink(src, lamb):
        return torch.sgn(src) * torch.max(torch.abs(src) - lamb, torch.zeros_like(src))

    def conj_grad(AHA, AHy, x_init, max_iter: int = 6, tol: float = 0.):

        # x = torch.zeros_like(AHy)
        # i, r, p = 0, AHy, AHy

        x = x_init.clone() 
        i, r, p = 0, AHy - AHA(x), AHy - AHA(x) 
        
        rTr = torch.sum(r.conj()*r).real
        while i < max_iter and rTr > 1e-10:
            Ap = AHA(p)
            alpha = rTr / torch.sum(p.conj()*Ap).real
            alpha = alpha
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r.conj()*r).real
            beta = rTrNew / rTr
            beta = beta
            p = r + beta * p
            i += 1
            rTr = rTrNew
        return x

    def CS_routine(x_init, t, niter=20):  
        withMoDL, withADMM = False, True

        x = x_init.clone()
        v = x_init.clone() 
        u = torch.zeros_like(x_init)
                                                              
        for i in range(niter): 
                
            d, s, h, w = x.shape 
                              
            if withSense and withADMM:
                AHA = lambda x: Sense.adjoint(Sense(x.view(1, d, 1, 1, s, h, w))).view(x.shape) + rho * x  
                AHy = ATy + rho * (v - u)
                # update x
                x = conj_grad(AHA, AHy, x, max_iter=6).to(dtype=torch.complex64)  # (d, s, h, w)
                v = torch.view_as_complex(shrink(torch.view_as_real(solve_operator(x, t) + u), lamb_1 / rho))
                u = u + x - v
                
            elif withSense and withMoDL:
                # x = T(x).float()
                # x = T.adjoint(padding(NN_Module(crop(x, crop_size=192)), pad_size=4)).squeeze()
                AHA = lambda x: Sense.adjoint(Sense(x.view(1, d, 1, 1, s, h, w))).view(x.shape) + rho * x  
                AHy = ATy + rho * x
                x = conj_grad(AHA, AHy, x, max_iter=6).to(dtype=torch.complex64)               
            else:
                AHA = lambda x: _AT(_A(x)) + rho * ((x))  
                AHy = ATy + rho * (v - u)
        
                # update x, v, u
                x = conj_grad(AHA, AHy, x, max_iter=3).to(dtype=torch.complex64)   # max_iter=6                    
                v = torch.view_as_complex(shrink(torch.view_as_real(x + u), lamb_1 / rho))  
                # v = torch.view_as_complex(shrink(torch.view_as_real(solve_operator(x, t) + u), lamb_1 / rho))          
                u = x - v + u

                    
        x = kspace_to_nchw(torch.view_as_real(x).permute(1, 0, 2, 3, 4).contiguous()) # (s, 2 * d, h, w)                    
        x_mean = x
        return x, x_mean, v

    def get_update_fn(update_fn, model):
        def radon_update_fn(x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, t):
            with torch.no_grad():
                x = torch.view_as_complex(nchw_to_kspace(x).permute(1, 0, 2, 3, 4).contiguous()) # (d, s, h, w)
                
                if Sense_recon:
                    x, x_mean, v = CS_routine(x, t, niter=20) 
                else:
                    x, x_mean, v = CS_routine(x, t, niter=1)
                                       
                return x, x_mean, v
        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn, model)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn, model)
    mc_update_fn = get_ADMM_TV_fn()

    def solve_operator(x, t): # x : (d, s, h, w)
        patch_size, image_size = 48, 192
        encoder = PatchEncoder(patch_size, image_size)
        x = kspace_to_nchw(torch.view_as_real(x).permute(1, 0, 2, 3, 4).contiguous()) # (s, 2 * d, h, w) 
        if shepp_logan: # (s, 2 * d, h, w) --> (b, 2 * p, h', w')
            x = crop(x, crop_size=192) 
            x = nchw_to_kspace(x)  # (s, d, 192, 192, 2)  
            x = x.permute(1, 0, 2, 3, 4).contiguous() # (d, s, 192, 192, 2)  
            x = x.reshape(shape=(-1, 1, x.shape[-3], x.shape[-2], 2)) 
            x = encoder(x)  # (b, p, 48, 48, 2)
            x = kspace_to_nchw(x) 
            
        # 1. batchify into sizes that fit into the GPU              
        x_batch = batchfy(x, 20)
        # 2. Run PC step for each batch
        x_agg = list()
        for idx, x_batch_sing in enumerate(x_batch):

            # x_batch_sing: (3, 6, 200, 200)
            x_batch_sing, _ = predictor_denoise_update_fn(x_batch_sing, t)
            x_batch_sing, _ = corrector_denoise_update_fn(x_batch_sing, t)                   
            x_agg.append(x_batch_sing)
            
        # 3. Aggregate to run ADMM TV
        x = torch.cat(x_agg, dim=0)
        if shepp_logan:  # (b, 2 * p, h', w') --> (s, 2 * d, h, w) 
            x = nchw_to_kspace(x)  # (b, p, h', w', 2)
            x = encoder.backward(x) # (b, 1, 192, 192, 2)
            x = torch.view_as_complex(x) 
            x = padding(x, pad_size=4) 
            x = x.reshape(shape=(-1, 3, x.shape[-2], x.shape[-1])) # (d, s, h, w)
            x = torch.view_as_real(x)   # (d, s, h, w, 2)
            x = x.permute(1, 0, 2, 3, 4).contiguous() 
            x = kspace_to_nchw(x) 
        x = torch.view_as_complex(nchw_to_kspace(x).permute(1, 0, 2, 3, 4).contiguous()) # (d, s, h, w)
        return x
               
    def pc_radon():
        save_root = '/home/woody/iwbi/iwbi102h/diffusion_model_study/figures'
        save_root = Path(save_root)
        (save_root / 'recon_steps').mkdir(parents=True, exist_ok=True)
                     
        collapse_slices, num_diffusions, image_size = sizes
        data = torch.zeros((collapse_slices, 2 * num_diffusions, image_size, image_size))  # (s, 2 * d, h, w) 
            
        if shepp_logan:  # (b, 2 * p, h', w') 
            batch_size, patch_size, image_size = sizes
            patches = (image_size // patch_size)**2
            encoder = PatchEncoder(patch_size, image_size)
            data = torch.zeros((batch_size, 2 * patches, patch_size, patch_size))  # (b, 2 * p, h', w')                
                   
        with torch.no_grad():           
            x = sde.prior_sampling(data.shape).to(device) 
            timesteps = torch.linspace(sde.T, eps, sde.N)
            # x = kspace_to_nchw(torch.view_as_real(ATy).permute(1, 0, 2, 3, 4).contiguous()) # (s, 2 * d, h, w)              
                                                   
            recon_img = ATy
            d, s = 0, 0
            recon_img_ds = normalize(recon_img[d][s])
            plt.imsave(save_root / 'recon_steps' / f'start_{d}{s}.png', clear(torch.abs(recon_img_ds), normalize=False), cmap='gray', vmin=0, vmax=torch.amax(torch.abs(recon_img_ds)) * 0.6)
        
            for i in tqdm(range(sde.N)):  # int(0.7 * sde.N),                 
                t = timesteps[i]
                
                if solve_in_admm is not True:
                    # 1. batchify into sizes that fit into the GPU
                    x_batch = batchfy(x, 20)
                    # 2. Run PC step for each batch
                    x_agg = list()
                    for idx, x_batch_sing in enumerate(x_batch):
                        x_batch_sing, _ = predictor_denoise_update_fn(x_batch_sing, t)
                        x_batch_sing, _ = corrector_denoise_update_fn(x_batch_sing, t)                   
                        x_agg.append(x_batch_sing)    
                    # 3. Aggregate to run ADMM TV
                    x = torch.cat(x_agg, dim=0)
                              
                if shepp_logan:  # (b, 2 * p, h', w') --> (s, 2 * d, h, w) 
                    x = nchw_to_kspace(x)  # (b, p, h', w', 2)
                    x = encoder.backward(x) # (b, 1, 192, 192, 2)
                    x = torch.view_as_complex(x) 
                    x = padding(x, pad_size=4) 
                    x = x.reshape(shape=(-1, 3, x.shape[-2], x.shape[-1])) # (d, s, h, w)
                    x = torch.view_as_real(x)   # (d, s, h, w, 2)
                    x = x.permute(1, 0, 2, 3, 4).contiguous() 
                    x = kspace_to_nchw(x)     
                                                          
                # 4. Run ADMM TV
                x, x_mean, v_solve = mc_update_fn(x, t) # (s, 2 * d, h, w)
                
                if (i % 50 == 0):
                    recon_img = nchw_to_kspace(x)  # (s, d, 192, 192, 2)  
                    recon_img = recon_img.permute(1, 0, 2, 3, 4).contiguous() # (d, s, 192, 192, 2)
                    recon_img = torch.view_as_complex(recon_img) 
                    recon_img_ds = normalize(recon_img[d][s])
                    v_solve_ds = normalize(v_solve[d][s])
                    # plt.imsave(save_root / 'recon_steps' / f'solve_{i}.png', clear(torch.abs(v_solve_ds), normalize=False), cmap='gray', vmin=0, vmax=torch.amax(torch.abs(v_solve_ds)) * 0.6)            
                    plt.imsave(save_root / 'recon_steps' / f'admm_{i}.png', clear(torch.abs(recon_img_ds), normalize=False), cmap='gray', vmin=0, vmax=torch.amax(torch.abs(recon_img_ds)) * 0.6)
                               
                if shepp_logan: # (s, 2 * d, h, w) --> (b, 2 * p, h', w')
                    x = crop(x, crop_size=192) 
                    x = nchw_to_kspace(x)  # (s, d, 192, 192, 2)  
                    x = x.permute(1, 0, 2, 3, 4).contiguous() # (d, s, 192, 192, 2)  
                    x = x.reshape(shape=(-1, 1, x.shape[-3], x.shape[-2], 2)) 
                    x = encoder(x)  # (b, p, 48, 48, 2)
                    x = kspace_to_nchw(x) 
                                    
                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[0:1]), cmap='gray')

            if shepp_logan:  # (b, 2 * p, h', w') --> (s, 2 * d, h, w)
                x = nchw_to_kspace(x)  #  (b, p, h', w', 2)
                x = encoder.backward(x) #  (b, 1, 192, 192, 2)
                x = torch.view_as_complex(x) 
                x = padding(x, pad_size=4) 
                x = x.reshape(shape=(-1, 3, x.shape[-2], x.shape[-1])) # (d, s, h, w)
                x = torch.view_as_real(x)
                x = x.permute(1, 0, 2, 3, 4).contiguous() 
                x = kspace_to_nchw(x) 

            x = torch.view_as_complex(nchw_to_kspace(x).permute(1, 0, 2, 3, 4).contiguous())            
            x_mean = x 
                           
            return inverse_scaler(x_mean if denoise else x) 

    return pc_radon